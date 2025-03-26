#include "dd/DesicionDiagramContainer.hpp"

namespace dd {

void DDContainerBase::reset() {
  ut.clear();
  mm.reset();
}

bool DDContainerBase::garbageCollect(bool force) {
  if (!force && !ut.possiblyNeedsCollection()) {
    return false;
  }

  return ut.garbageCollect(force) > 0;
}

void DDContainerBase::addStatsJson(nlohmann::basic_json<>& j,
                                   bool includeIndividualTables) const {
  j["unique_table"] = ut.getStatsJson(includeIndividualTables);
  j["memory_manager"] = mm.getStats().json();
}

} // namespace dd