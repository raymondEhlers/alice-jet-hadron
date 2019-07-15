
#include <iostream>

// Tell ROOT where to find AliRoot headers
R__ADD_INCLUDE_PATH($ALICE_ROOT)
// Tell ROOT where to find AliPhysics headers
R__ADD_INCLUDE_PATH($ALICE_PHYSICS)
#include <AliAnalysisTaskEmcalJetHUtils.h>
class JetHUtilsPublic : public PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils {
 public:
  // LHC15o
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_0_10_pt;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_10_30_pt;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_30_50_pt;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_50_90_pt;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_0_10_eta;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_10_30_eta;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_30_50_eta;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_50_90_eta;
  // And the related functions
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oLowPtEfficiency;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oHighPtEfficiency;
  using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oEtaEfficiency;
};

void test_moving_public() {
  std::cout << "val: " << JetHUtilsPublic::LHC15oParam_0_10_pt[0] << "\n";
  // This works... It appears not to work in PyROOT due to a bug.
}
