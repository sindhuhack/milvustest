#include "utils/Json.h"
#include "segcore/SegmentBase.h"
#include "query/generated/ExecPlanNodeVisitor.h"
#include "segcore/SegmentSmallIndex.h"

namespace milvus::query {

#if 1
namespace impl {
// THIS CONTAINS EXTRA BODY FOR VISITOR
// WILL BE USED BY GENERATOR UNDER suvlim/core_gen/
class ExecPlanNodeVisitor : PlanNodeVisitor {
 public:
    using RetType = segcore::QueryResult;
    ExecPlanNodeVisitor(segcore::SegmentBase& segment, segcore::Timestamp timestamp, const float* src_data)
        : segment_(segment), timestamp_(timestamp), src_data_(src_data) {
    }
    // using RetType = nlohmann::json;

    RetType get_moved_result(PlanNode& node){
        assert(!ret_.has_value());
        node.accept(*this);
        assert(ret_.has_value());
        auto ret = std::move(ret_).value();
        ret_ = std::nullopt;
        return ret;
    }
 private:
    // std::optional<RetType> ret_;
    segcore::SegmentBase& segment_;
    segcore::Timestamp timestamp_;
    const float* src_data_;

    std::optional<RetType> ret_;
};
}  // namespace impl
#endif

void
ExecPlanNodeVisitor::visit(FloatVectorANNS& node) {
    // TODO: optimize here, remove the dynamic cast
    assert(!ret_.has_value());
    auto segment = dynamic_cast<segcore::SegmentSmallIndex*>(&segment_);
    AssertInfo(segment, "support SegmentSmallIndex Only");
    RetType ret;
    segment->QueryBruteForceImpl(node.query_info_, src_data_, timestamp_, ret);
    ret_ = ret;
}

void
ExecPlanNodeVisitor::visit(BinaryVectorANNS& node) {
    // TODO
}

}  // namespace milvus::query
