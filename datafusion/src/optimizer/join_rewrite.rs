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
use crate::logical_plan::{col, Column, DFSchema, DFSchemaRef, JoinType, Operator};
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

/// converts "A AND B AND C" => [A, B, C]
fn split_members(predicate: &Expr) -> Vec<&Expr> {
    match predicate {
        Expr::BinaryExpr {
            right,
            op: Operator::And,
            left,
        } => {
            let mut exprs = vec![];
            exprs.append(&mut split_members(left));
            exprs.append(&mut split_members(right));
            exprs
        }
        Expr::Alias(expr, _) => split_members(expr),
        other => vec![other],
    }
}

fn get_schema_columns(schema: &DFSchema) -> HashSet<Column> {
    schema
        .fields()
        .iter()
        .map(|f| {
            [
                f.qualified_column(),
                // we need to push down filter using unqualified column as well
                f.unqualified_column(),
            ]
        })
        .flatten()
        .collect::<HashSet<_>>()
}

fn join_side_has_null_removing_filter(
    predicates: &Vec<&Expr>,
    side: &LogicalPlan,
) -> bool {
    let side_columns = get_schema_columns(side.schema());

    let mut null_removing_filters = vec![];
    predicates
        .into_iter()
        .try_for_each::<_, Result<()>>(|predicate| {
            let mut predicate_columns: HashSet<Column> = HashSet::new();
            utils::expr_to_columns(predicate, &mut predicate_columns)?;

            let predicate_only_references_side =
                predicate_columns.is_subset(&side_columns);
            let predicate_is_null_removing = expr_removes_nulls(
                predicate,
                side.schema(),
                &ExecutionContextState::new(),
            )?;
            if predicate_only_references_side && predicate_is_null_removing {
                null_removing_filters.push(predicate);
            }

            Ok(())
        });

    null_removing_filters.len() > 0
}

fn rewrite_join(filter: &Filter, join: &Join) -> Result<LogicalPlan> {
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

    let all_predicates = split_members(&predicate);

    let left_has_null_removing_filter =
        join_side_has_null_removing_filter(&all_predicates, left);
    let right_has_null_removing_filter =
        join_side_has_null_removing_filter(&all_predicates, right);

    let new_join_type = match join.join_type {
        JoinType::Right if right_has_null_removing_filter => JoinType::Inner,
        JoinType::Left if left_has_null_removing_filter => JoinType::Inner,
        JoinType::Full
            if right_has_null_removing_filter && left_has_null_removing_filter =>
        {
            JoinType::Inner
        }
        JoinType::Full if left_has_null_removing_filter => JoinType::Right,
        JoinType::Full if right_has_null_removing_filter => JoinType::Left,
        jt => jt,
    };

    Ok(LogicalPlan::Filter(Filter {
        predicate: predicate.clone(),
        input: Arc::new(LogicalPlan::Join(Join {
            left: left.clone(),
            right: right.clone(),
            on: on.clone(),
            join_type: new_join_type,
            join_constraint: join_constraint.clone(),
            schema: schema.clone(),
            null_equals_null: null_equals_null.clone(),
        })),
    }))
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
                return rewrite_join(filter, join);
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
    use crate::logical_plan::{col, exp, DFField, DFSchema};
    use crate::physical_plan::expressions::is_null;
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
}
