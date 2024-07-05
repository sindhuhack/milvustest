#pragma once

#include <fmt/core.h>

#include "common/Vector.h"
#include "exec/expression/Expr.h"
#include "expr/ITypeExpr.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

class PhyGISFunctionFilterExpr : public SegmentExpr {
 public:
    PhyGISFunctionFilterExpr(
        const std::vector<std::shared_ptr<Expr>>& input,
        const std::shared_ptr<const milvus::expr::GISFunctioinFilterExpr>& expr,
        const std::string& name,
        const segcore::SegmentInternalInterface* segment,
        int64_t active_count,
        int64_t batch_size)
        : SegmentExpr(std::move(input),
                      name,
                      segment,
                      expr->column_.field_id_,
                      active_count,
                      batch_size),
          expr_(expr) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override;

 private:
    //  VectorPtr
    //  EvalForIndexSegment();

    VectorPtr
    EvalForDataSegment();

 private:
    std::shared_ptr<const milvus::expr::GISFunctioinFilterExpr> expr_;
};
}  //namespace exec
}  // namespace milvus
