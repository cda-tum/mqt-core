configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(QUANTUM_OPT_TEST_DEPENDS FileCheck count not quantum-opt)

add_lit_testsuite(check-quantum-opt "Running the quantum-opt regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${QUANTUM_OPT_TEST_DEPENDS})
set_target_properties(check-quantum-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(QUANTUM_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${QUANTUM_OPT_TEST_DEPENDS})
