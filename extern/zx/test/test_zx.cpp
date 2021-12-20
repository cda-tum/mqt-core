#include <gtest/gtest.h>
//#include "..Hexagram.hpp"
#include "../include/ZXDiagram.hpp"

class ZXDiagramTest : public ::testing::Test {
public:
  zx::ZXDiagram diag;

  /*
   * Diagram should look like this:
   * {0: (Boundary, Phase=0)} ---H--- {2: (Z, Phase=0)} ---- {3: (Z, Phase = 0)} ----- {5: (Boundary, Phase = 0)}
   *                                                                 |
   *                                                                 |
   *                                                                 | 
   * {1: (Boundary, Phase=0)} ------------------------------ {4: (X, Phase = 0)} ----- {6: (Boundary, Phase = 0)}
*/
protected:
  virtual void SetUp() {
    diag = zx::ZXDiagram("./test/circuits/bell_state.qasm");
  }
};

  
TEST_F(ZXDiagramTest, parse_qasm) {
  EXPECT_EQ(diag.get_nvertices(), 7);
  EXPECT_EQ(diag.get_nedges(), 6);

  auto inputs = diag.get_inputs();
  EXPECT_EQ(inputs[0], 0);
  EXPECT_EQ(inputs[1], 1);

  auto outputs = diag.get_outputs();
  EXPECT_EQ(outputs[0], 5);
  EXPECT_EQ(outputs[1], 6);

  EXPECT_EQ(diag.get_edge(0, 2).value().type, zx::EdgeType::Hadamard);
  EXPECT_EQ(diag.get_edge(3, 4).value().type, zx::EdgeType::Simple);

  EXPECT_EQ(diag.get_vdata(0).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(1).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(2).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(3).value().type, zx::VertexType::Z);
  EXPECT_EQ(diag.get_vdata(4).value().type, zx::VertexType::X);
  EXPECT_EQ(diag.get_vdata(5).value().type, zx::VertexType::Boundary);
  EXPECT_EQ(diag.get_vdata(6).value().type, zx::VertexType::Boundary);

  for(auto i = 0; i < diag.get_nvertices(); i++)
    EXPECT_EQ(diag.get_vdata(6).value().phase, 0);
}

TEST_F(ZXDiagramTest, deletions) {
  diag.remove_vertex(3);
  EXPECT_EQ(diag.get_nvertices(), 6);
  EXPECT_EQ(diag.get_nedges(), 3);
  EXPECT_FALSE(diag.get_vdata(3).has_value());

  diag.remove_edge(0, 2);
  EXPECT_EQ(diag.get_nvertices(), 6);
  EXPECT_EQ(diag.get_nedges(), 2);
}




int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
