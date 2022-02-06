// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Join rewrite optimizer rule

use crate::error::Result;
use crate::execution::context::{ExecutionContextState, ExecutionProps};
use crate::logical_plan::plan::{Filter, Join};
use crate::logical_plan::{Column, DFSchemaRef, JoinType};
use crate::logical_plan::{Expr, LogicalPlan};
use crate::optimizer::optimizer::OptimizerRule;
use crate::optimizer::utils;
use crate::optimizer::utils::optimize_children;
use crate::physical_plan::planner::DefaultPhysicalPlanner;
use crate::physical_plan::ColumnarValue;
use crate::scalar::ScalarValue;
use arrow::array::{new_null_array, Array};
use arrow::record_batch::RecordBatch;
use std::collections::HashSet;
use std::sync::Arc;

/// Optimizer that takes advantage of predicates that filter nulls out to rewrite outer joins in a
/// more efficient way. For example, in the following query the t2.y IS NULL filter removes all
/// possible null rows produced by the left join:
///
///   SELECT t1.x, t2.y
///   FROM t1 LEFT JOIN t2 ON t1.x = t2.y
///   WHERE t2.y IS NULL
///
/// Thus, this query is rewritten as:
///
///   SELECT t1.x, t2.y
///   FROM t1 INNER JOIN t2 ON t1.x = t2.y
///   WHERE t2.y IS NULL
///
/// Rewriting joins in this fashion enables more efficient join algorithms and further optimizations
/// to the plan (e.g. - pushing predicates past an inner join).
#[derive(Default)]
pub struct JoinRewrite {}

impl JoinRewrite {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

fn expr_removes_nulls(
    e: &Expr,
    input_schema: &DFSchemaRef,
    state: &ExecutionContextState,
) -> Result<bool> {
    let schema = input_schema.as_ref().to_owned().into();
    let planner = DefaultPhysicalPlanner::with_extension_planners(vec![]);
    let physical_expr = planner.create_physical_expr(&e, input_schema, &schema, state)?;

    let null_columns = schema
        .fields()
        .iter()
        .map(|field| new_null_array(field.data_type(), 1))
        .collect();
    let null_batch = RecordBatch::try_new(Arc::new(schema), null_columns)?;

    let val = physical_expr.evaluate(&null_batch)?;

    let scalar_removes_null = |v: &ScalarValue| match v {
        ScalarValue::Boolean(Some(false)) => Ok(true),
        _ => Ok(v.is_null()),
    };

    if let ColumnarValue::Array(v) = val {
        assert_eq!(v.len(), 1);
        let v = ScalarValue::try_from_array(&v, 0)?;
        scalar_removes_null(&v)
    } else {
        unreachable!()
    }
}

fn join_side_has_null_removing_filter(
    predicates: &Vec<&Expr>,
    side: &LogicalPlan,
) -> bool {
    let side_columns = utils::get_all_columns_for_schema(side.schema().clone());

    let mut null_removing_filters = vec![];
    predicates
        .into_iter()
        .try_for_each::<_, Result<()>>(|predicate| {
            let mut predicate_columns: HashSet<Column> = HashSet::new();
            utils::expr_to_columns(predicate, &mut predicate_columns)?;
            if predicate_columns.is_subset(&side_columns)
                && expr_removes_nulls(
                    predicate,
                    side.schema(),
                    &ExecutionContextState::new(),
                )?
            {
                null_removing_filters.push(predicate);
            }

            Ok(())
        });

    null_removing_filters.len() > 0
}

fn rewrite_join(filter: &Filter, join: &Join) -> Result<Join> {
    let Filter { predicate, .. } = filter;
    let Join {
        left,
        right,
        on,
        join_type,
        join_constraint,
        schema,
        null_equals_null,
        ..
    } = join;

    let mut all_predicates = vec![];
    utils::split_members(&predicate, &mut all_predicates);

    let left_has_null_removing_filter =
        join_side_has_null_removing_filter(&all_predicates, left);
    let right_has_null_removing_filter =
        join_side_has_null_removing_filter(&all_predicates, right);

    let new_join_type = match join_type {
        JoinType::Right if left_has_null_removing_filter => JoinType::Inner,
        JoinType::Left if right_has_null_removing_filter => JoinType::Inner,
        JoinType::Full
            if right_has_null_removing_filter && left_has_null_removing_filter =>
        {
            JoinType::Inner
        }
        JoinType::Full if left_has_null_removing_filter => JoinType::Left,
        JoinType::Full if right_has_null_removing_filter => JoinType::Right,
        jt => jt.clone(),
    };

    Ok(Join {
        left: left.clone(),
        right: right.clone(),
        on: on.clone(),
        join_type: new_join_type,
        join_constraint: join_constraint.clone(),
        schema: schema.clone(),
        null_equals_null: null_equals_null.clone(),
    })
}

impl OptimizerRule for JoinRewrite {
    fn optimize(
        &self,
        plan: &LogicalPlan,
        execution_props: &ExecutionProps,
        execution_state: &ExecutionContextState,
    ) -> Result<LogicalPlan> {
        if let LogicalPlan::Filter(filter) = plan {
            if let LogicalPlan::Join(join) = filter.input.as_ref() {
                let join = LogicalPlan::Join(rewrite_join(filter, join)?);
                let input =
                    optimize_children(self, &join, execution_props, execution_state)?;
                return Ok(LogicalPlan::Filter(Filter {
                    predicate: filter.predicate.clone(),
                    input: Arc::new(input),
                }));
            }
        }

        optimize_children(self, plan, execution_props, execution_state)
    }
    fn name(&self) -> &str {
        "JoinRewrite"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::{and, col, DFField, DFSchema, LogicalPlanBuilder};
    use crate::test::{test_table_scan, test_table_scan_with_name};
    use arrow::datatypes::{DataType, Field};

    fn check_expr_removes_nulls_output(expr: &Expr, expected: bool) -> Result<()> {
        let schema = Arc::new(DFSchema::new(vec![
            DFField::from(Field::new("a", DataType::UInt32, true)),
            DFField::from(Field::new("b", DataType::UInt32, true)),
            DFField::from(Field::new("c", DataType::Utf8, true)),
            DFField::from(Field::new("d", DataType::Boolean, true)),
        ])?);

        assert_eq!(
            expr_removes_nulls(expr, &schema, &ExecutionContextState::new())?,
            expected
        );

        Ok(())
    }

    #[test]
    fn expr_removes_nulls_col() -> Result<()> {
        let expr = col("a");
        check_expr_removes_nulls_output(&expr, true)
    }

    #[test]
    fn expr_removes_nulls_col_eq_col() -> Result<()> {
        let expr = col("a").eq(col("b"));
        // null = null evaluates to null
        check_expr_removes_nulls_output(&expr, true)
    }

    #[test]
    fn expr_removes_nulls_col_is_null() -> Result<()> {
        let expr = col("a").is_null();
        check_expr_removes_nulls_output(&expr, false)
    }

    #[test]
    fn expr_removes_nulls_col_is_not_null() -> Result<()> {
        let expr = col("a").is_not_null();
        check_expr_removes_nulls_output(&expr, true)
    }

    fn assert_optimized_plan_eq(plan: &LogicalPlan, expected: &str) {
        let rule = JoinRewrite::new();
        let optimized_plan = rule
            .optimize(plan, &ExecutionProps::new(), &ExecutionContextState::new())
            .expect("failed to optimize plan");
        let formatted_plan = format!("{:?}", optimized_plan);
        assert_eq!(formatted_plan, expected);
    }

    fn construct_join_plan(join_type: JoinType) -> Result<LogicalPlanBuilder> {
        let table_scan = test_table_scan()?;
        let left = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("c")])?
            .build()?;
        let right_table_scan = test_table_scan_with_name("test2")?;
        let right = LogicalPlanBuilder::from(right_table_scan)
            .project(vec![col("a"), col("b")])?
            .build()?;
        LogicalPlanBuilder::from(left).join(
            &right,
            join_type,
            (vec![Column::from_name("a")], vec![Column::from_name("a")]),
        )
    }

    #[test]
    fn join_rewrite_right_to_inner() -> Result<()> {
        let plan = construct_join_plan(JoinType::Right)?
            .filter(col("c").is_not_null())?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test.c IS NOT NULL\
            \n  Right Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        let expected = "\
            Filter: #test.c IS NOT NULL\
            \n  Inner Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None";

        assert_optimized_plan_eq(&plan, expected);
        Ok(())
    }

    #[test]
    fn join_rewrite_left_to_inner() -> Result<()> {
        let plan = construct_join_plan(JoinType::Left)?
            .filter(col("b").is_not_null())?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test2.b IS NOT NULL\
            \n  Left Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        let expected = "\
            Filter: #test2.b IS NOT NULL\
            \n  Inner Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None";

        assert_optimized_plan_eq(&plan, expected);
        Ok(())
    }

    #[test]
    fn join_rewrite_full_to_right() -> Result<()> {
        let plan = construct_join_plan(JoinType::Full)?
            .filter(col("b").is_not_null())?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test2.b IS NOT NULL\
            \n  Full Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        let expected = "\
            Filter: #test2.b IS NOT NULL\
            \n  Right Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None";

        assert_optimized_plan_eq(&plan, expected);
        Ok(())
    }

    #[test]
    fn join_rewrite_full_to_left() -> Result<()> {
        let plan = construct_join_plan(JoinType::Full)?
            .filter(col("c").is_not_null())?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test.c IS NOT NULL\
            \n  Full Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        let expected = "\
            Filter: #test.c IS NOT NULL\
            \n  Left Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None";

        assert_optimized_plan_eq(&plan, expected);
        Ok(())
    }

    #[test]
    fn join_rewrite_full_to_inner() -> Result<()> {
        let plan = construct_join_plan(JoinType::Full)?
            .filter(and(col("b").is_not_null(), col("c").is_not_null()))?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test2.b IS NOT NULL AND #test.c IS NOT NULL\
            \n  Full Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        let expected = "\
            Filter: #test2.b IS NOT NULL AND #test.c IS NOT NULL\
            \n  Inner Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None";

        assert_optimized_plan_eq(&plan, expected);
        Ok(())
    }

    #[test]
    fn join_rewrite_right_noop() -> Result<()> {
        let plan = construct_join_plan(JoinType::Right)?
            // The filter on the left side does not remove nulls.
            .filter(and(col("c").is_null(), col("b").is_not_null()))?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test.c IS NULL AND #test2.b IS NOT NULL\
            \n  Right Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        // We expect a noop here - i.e. the optimized plan == the unoptimized plan.
        assert_optimized_plan_eq(&plan, &format!("{:?}", plan));
        Ok(())
    }

    #[test]
    fn join_rewrite_left_noop() -> Result<()> {
        let plan = construct_join_plan(JoinType::Left)?
            // The filter on the right side does not remove nulls.
            .filter(and(col("c").is_not_null(), col("b").is_null()))?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test.c IS NOT NULL AND #test2.b IS NULL\
            \n  Left Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        // We expect a noop here - i.e. the optimized plan == the unoptimized plan.
        assert_optimized_plan_eq(&plan, &format!("{:?}", plan));
        Ok(())
    }

    #[test]
    fn join_rewrite_full_noop() -> Result<()> {
        let plan = construct_join_plan(JoinType::Full)?
            // Neither of the filters remove nulls.
            .filter(and(col("c").is_null(), col("b").is_null()))?
            .build()?;

        // not part of the test, just good to know:
        assert_eq!(
            format!("{:?}", plan),
            "\
            Filter: #test.c IS NULL AND #test2.b IS NULL\
            \n  Full Join: #test.a = #test2.a\
            \n    Projection: #test.a, #test.c\
            \n      TableScan: test projection=None\
            \n    Projection: #test2.a, #test2.b\
            \n      TableScan: test2 projection=None"
        );

        // We expect a noop here - i.e. the optimized plan == the unoptimized plan.
        assert_optimized_plan_eq(&plan, &format!("{:?}", plan));
        Ok(())
    }
}
