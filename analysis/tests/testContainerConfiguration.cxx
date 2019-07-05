/**
 * Tests for EMCal container configuration through external configuration functions using YAML.
 *
 * @author: Raymond Ehlers, Yale University <raymond.ehlers@cern.ch>
 * @date: 4 July 2019
 */

#include <AliLog.h>
#include <AliVCluster.h>

#include "AliYAMLConfiguration.h"
#include "AliEmcalContainer.h"
#include "AliParticleContainer.h"
#include "AliTrackContainer.h"
#include "AliClusterContainer.h"
#include "AliEmcalTrackSelection.h"

#include "AliAnalysisTaskEmcalJetHUtils.h"

double epsilon = 0.0001;

/**
 * Test that we return a basic (unconfigured) particle container.
 *
 * @param[in] yamlConfig YAML configuration.
 */
bool testBasicParticleContainer(PWG::Tools::AliYAMLConfiguration & yamlConfig)
{
  AliParticleContainer* partCont = new AliParticleContainer("tracks");
  std::string containerName = "basic";
  std::vector<std::string> baseNameWithContainer =
    std::vector<std::string>({"particle", containerName});
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureEMCalContainersFromYAMLConfig(
    baseNameWithContainer, containerName, partCont, yamlConfig, "test");

  // Check the values
  // In this basic check, we rely heavily on the default values of AliEmcalContainer.
  assert(TClass::GetClass(typeid(*partCont)) == TClass::GetClass("AliParticleContainer"));
  assert(std::string(partCont->GetName()) == "basic");
  assert(std::abs(partCont->GetMinPt() - 0.15) < epsilon);
  assert(std::abs(partCont->GetMinE() - 0.0) < epsilon);
  assert(std::abs(partCont->GetMinEta() - (-0.9)) < epsilon);
  assert(std::abs(partCont->GetMaxEta() - (0.9)) < epsilon);
  assert(std::abs(partCont->GetMinPhi() - (-10)) < epsilon);
  assert(std::abs(partCont->GetMaxPhi() - (10)) < epsilon);
  assert(partCont->GetIsEmbedding() == false);

  return true;
}

/**
 * Test that a particle container is correctly configured.
 *
 * @param[in] yamlConfig YAML configuration.
 */
bool testParticleContainer(PWG::Tools::AliYAMLConfiguration & yamlConfig)
{
  AliParticleContainer * partCont = new AliParticleContainer("tracks");
  std::string containerName = "configured";
  std::vector<std::string> baseNameWithContainer =
    std::vector<std::string>({"particle", containerName});

  // Configure the container
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureEMCalContainersFromYAMLConfig(
    baseNameWithContainer, containerName, partCont, yamlConfig, "test");

  // Check the values
  assert(TClass::GetClass(typeid(*partCont)) == TClass::GetClass("AliParticleContainer"));
  assert(std::string(partCont->GetName()) == "configured");
  assert(std::abs(partCont->GetMinPt() - 1.0) < epsilon);
  assert(std::abs(partCont->GetMinE() - 2.2) < epsilon);
  assert(std::abs(partCont->GetMinEta() - (-0.3)) < epsilon);
  assert(std::abs(partCont->GetMaxEta() - (0.4)) < epsilon);
  assert(std::abs(partCont->GetMinPhi() - (0)) < epsilon);
  assert(std::abs(partCont->GetMaxPhi() - (3.14)) < epsilon);
  assert(partCont->GetIsEmbedding() == true);

  return true;
}

/**
 * Test that we return a basic (unconfigured) track container.
 *
 * @param[in] yamlConfig YAML configuration.
 */
bool testBasicTrackContainer(PWG::Tools::AliYAMLConfiguration & yamlConfig)
{
  AliTrackContainer * trackCont = new AliTrackContainer("tracks");
  std::string containerName = "basic";
  std::vector<std::string> baseNameWithContainer =
    std::vector<std::string>({"track", containerName});

  // Configure the container
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureEMCalContainersFromYAMLConfig(
    baseNameWithContainer, containerName, trackCont, yamlConfig, "test");
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureTrackContainersFromYAMLConfig(
    baseNameWithContainer, trackCont, yamlConfig, "test");

  // Check the basic values
  assert(std::string(trackCont->GetName()) == "basic");
  assert(std::abs(trackCont->GetMinPt() - 0.15) < epsilon);
  assert(std::abs(trackCont->GetMinE() - 0.0) < epsilon);
  assert(TClass::GetClass(typeid(*trackCont)) == TClass::GetClass("AliTrackContainer"));
  // Check the particular track values
  // In this basic check, we rely heavily on the default values of AliEmcalContainer.
  assert(trackCont->GetAODFilterBits() == 0);
  assert(trackCont->GetTrackFilterType() == AliEmcalTrackSelection::kHybridTracks);
  // This function doesn't exist...
  //assert(std::string(trackCont->GetTrackCutsPeriod()) == "");

  return true;
}

/**
 * Test that a track container is correctly configured.
 *
 * @param[in] yamlConfig YAML configuration.
 */
bool testTrackContainer(PWG::Tools::AliYAMLConfiguration & yamlConfig)
{
  AliTrackContainer * trackCont = new AliTrackContainer("tracks");
  std::string containerName = "configured";
  std::vector<std::string> baseNameWithContainer =
    std::vector<std::string>({"track", containerName});

  // Configure the container
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureEMCalContainersFromYAMLConfig(
    baseNameWithContainer, containerName, trackCont, yamlConfig, "test");
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureTrackContainersFromYAMLConfig(
    baseNameWithContainer, trackCont, yamlConfig, "test");

  // Check the basic values
  assert(std::string(trackCont->GetName()) == "configured");
  assert(std::abs(trackCont->GetMinPt() - 2.0) < epsilon);
  assert(TClass::GetClass(typeid(*trackCont)) == TClass::GetClass("AliTrackContainer"));
  // Check the particular track values
  // 272 == 2^4 + 2^8
  assert(trackCont->GetAODFilterBits() == 272);
  assert(trackCont->GetTrackFilterType() == AliEmcalTrackSelection::kTPCOnlyTracks);
  // This function doesn't exist...
  //assert(std::string(trackCont->GetTrackCutsPeriod()) == "LHC15o");

  return true;
}

/**
 * Test that we return a basic (unconfigured) cluster container.
 *
 * @param[in] yamlConfig YAML configuration.
 */
bool testBasicClusterContainer(PWG::Tools::AliYAMLConfiguration & yamlConfig)
{
  AliClusterContainer * clusterCont = new AliClusterContainer("caloClusters");
  std::string containerName = "basic";
  std::vector<std::string> baseNameWithContainer =
    std::vector<std::string>({"cluster", containerName});

  // Configure the container
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureEMCalContainersFromYAMLConfig(
    baseNameWithContainer, containerName, clusterCont, yamlConfig, "test");
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureClusterContainersFromYAMLConfig(
    baseNameWithContainer, clusterCont, yamlConfig, "test");

  // Check the basic values
  assert(TClass::GetClass(typeid(*clusterCont)) == TClass::GetClass("AliClusterContainer"));
  assert(std::string(clusterCont->GetName()) == "basic");
  assert(std::abs(clusterCont->GetMinPt() - 0.15) < epsilon);
  // Explicitly check E because it's especially important for clusters
  assert(std::abs(clusterCont->GetMinE() - 0) < epsilon);
  // Check the particular cluster values
  // In this basic check, we rely heavily on the default values of AliEmcalContainer.
  assert(clusterCont->GetDefaultClusterEnergy() == -1);
  assert(std::abs(clusterCont->GetClusUserDefEnergyCut(AliVCluster::kNonLinCorr) - 0) < epsilon);
  assert(std::abs(clusterCont->GetClusUserDefEnergyCut(AliVCluster::kHadCorr) - 0) < epsilon);
  // This function doesn't exist...
  //assert(clusterCont->GetIncludePHOS() == false);

  return true;
}

/**
 * Test that a cluster container is correctly configured.
 *
 * @param[in] yamlConfig YAML configuration.
 */
bool testClusterContainer(PWG::Tools::AliYAMLConfiguration & yamlConfig)
{
  AliClusterContainer * clusterCont = new AliClusterContainer("caloClusters");
  std::string containerName = "configured";
  std::vector<std::string> baseNameWithContainer =
    std::vector<std::string>({"cluster", containerName});

  // Configure the container
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureEMCalContainersFromYAMLConfig(
    baseNameWithContainer, containerName, clusterCont, yamlConfig, "test");
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::ConfigureClusterContainersFromYAMLConfig(
    baseNameWithContainer, clusterCont, yamlConfig, "test");

  // Check the basic values
  assert(TClass::GetClass(typeid(*clusterCont)) == TClass::GetClass("AliClusterContainer"));
  assert(std::string(clusterCont->GetName()) == "configured");
  assert(std::abs(clusterCont->GetMinPt() - 3.0) < epsilon);
  assert(std::abs(clusterCont->GetMinE() - 2.2) < epsilon);
  // Check the particular cluster values
  // In this basic check, we rely heavily on the default values of AliEmcalContainer.
  assert(clusterCont->GetDefaultClusterEnergy() == AliVCluster::kNonLinCorr);
  assert(std::abs(clusterCont->GetClusUserDefEnergyCut(AliVCluster::kNonLinCorr) - 30.1) < epsilon);
  assert(std::abs(clusterCont->GetClusUserDefEnergyCut(AliVCluster::kHadCorr) - 50.2) < epsilon);
  // This function doesn't exist...
  //assert(clusterCont->GetIncludePHOS() == false);

  return true;
}

bool testContainerConfiguration()
{
  // Ensure that we can see the log messages.
  AliLog::SetClassDebugLevel("test", AliLog::kDebug + 0);
  // Setup the YAML configuration
  PWG::Tools::AliYAMLConfiguration yamlConfig;
  yamlConfig.AddConfiguration("test_configuration.yaml", "main");

  // Particle containers
  assert(testBasicParticleContainer(yamlConfig) == true);
  assert(testParticleContainer(yamlConfig) == true);
  // Track containers
  assert(testBasicTrackContainer(yamlConfig) == true);
  assert(testTrackContainer(yamlConfig) == true);
  // Cluster containers
  assert(testBasicClusterContainer(yamlConfig) == true);
  assert(testClusterContainer(yamlConfig) == true);

  return true;
}

