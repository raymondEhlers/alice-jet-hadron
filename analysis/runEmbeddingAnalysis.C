/// \file runJetHAnalysis.C
/// \brief Embedding run macro
///
/// \ingroup EMCALJETFW
/// Embedding run macro to create a response matrix
///
/// \author Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
/// \date Jul 27, 2016

class AliESDInputHandler;
class AliAODInputHandler;
class AliVEvent;
class AliAnalysisGrid;
class AliAnalysisManager;
class AliAnalysisAlien;
class AliPhysicsSelectionTask;
class AliCentralitySelectionTask;
class AliTaskCDBconnect;

class AliClusterContainer;
class AliParticleContainer;
class AliJetContainer;

class AliAnalysisTaskEmcalEmbeddingHelper;
class AliEmcalCorrectionTask;
class AliEmcalJetTask;
class AliAnalysisTaskEmcalJetSample;
class AliJetResponseMaker;

namespace PWGJE {
  namespace EMCALJetTasks {
    class AliEmcalJetTaggerTaskFast;
    class AliAnalysisTaskEmcalJetHPerformance;
    class AliAnalysisTaskEmcalJetHCorrelations;
  }
}

void LoadMacros();
void StartGridAnalysis(AliAnalysisManager* pMgr, const char* uniqueName, const char* cGridMode);
AliAnalysisGrid* CreateAlienHandler(const char* uniqueName, const char* gridDir, const char* gridMode, const char* runNumbers,
    const char* pattern, TString additionalCode, TString additionalHeaders, Int_t maxFilesPerWorker, Int_t workerTTL, Bool_t isMC);

#ifdef __CLING__
// Tell ROOT where to find AliRoot headers
R__ADD_INCLUDE_PATH($ALICE_ROOT)
// Tell ROOT where to find AliPhysics headers
R__ADD_INCLUDE_PATH($ALICE_PHYSICS)
// Simplify rho task usage
#include "PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C"
// Include AddTask to test for LEGO train
#include "PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetHCorrelations.C"
#include "PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetHPerformance.C"
#endif

AliAnalysisManager* runEmbeddingAnalysis(
    const char   *cDataType       = "AOD",                                   // set the analysis type, AOD or ESD
    const char   *cRunPeriod      = "LHC11h",                                // set the run period
    const char   *cEmbedRunPeriod = "LHC12a15e",                             // set the embedded run period
    const char   *cLocalFiles     = "aodFiles.txt",                          // set the local list file
    const UInt_t  iNumEvents      = 1000,                                    // number of events to be analyzed
    const UInt_t  kPhysSel        = AliVEvent::kEMCEGA | AliVEvent::kMB |
                    AliVEvent::kCentral | AliVEvent::kSemiCentral, //AliVEvent::kAny,                         // Physics selection
    const char   *cTaskName       = "EMCalEmbeddingAnalysis",                     // sets name of analysis manager
    // 0 = only prepare the analysis manager but do not start the analysis
    // 1 = prepare the analysis manager and start the analysis
    // 2 = launch a grid analysis
    Int_t         iStartAnalysis  = 1,
    const UInt_t  iNumFiles       = 5,                                     // number of files analyzed locally
    const char   *cGridMode       = "test"
)
{
  // Setup period
  TString runPeriod(cRunPeriod);
  runPeriod.ToLower();

  // Set Run 2
  Bool_t bIsRun2 = kFALSE;
  if (runPeriod.Length() == 6 && (runPeriod.BeginsWith("lhc15")
      || runPeriod.BeginsWith("lhc16")
      || runPeriod.BeginsWith("lhc17")
      || runPeriod.BeginsWith("lhc18"))) bIsRun2 = kTRUE;

  // Set beam type
  AliAnalysisTaskEmcal::BeamType iBeamType = AliAnalysisTaskEmcal::kpp;
  if (runPeriod == "lhc10h" || runPeriod == "lhc11h" || runPeriod == "lhc15o") {
    iBeamType = AliAnalysisTaskEmcal::kAA;
  }
  else if (runPeriod == "lhc12g" || runPeriod == "lhc13b" || runPeriod == "lhc13c" ||
      runPeriod == "lhc13d" || runPeriod == "lhc13e" || runPeriod == "lhc13f" ||
      runPeriod == "LHC16q" || runPeriod == "LHC16r" || runPeriod == "LHC16s" || runPeriod == "LHC16t") {
    iBeamType = AliAnalysisTaskEmcal::kpA;
  }

  // Ghost area
  Double_t kGhostArea = 0.01;
  if (iBeamType != AliAnalysisTaskEmcal::kpp) kGhostArea = 0.005;

  // Setup track container
  AliTrackContainer::SetDefTrackCutsPeriod(runPeriod);
  std::cout << "Default track cut period set to: " << AliTrackContainer::GetDefTrackCutsPeriod().Data() << "\n";

  // Set data file type
  enum eDataType { kAod, kEsd };

  eDataType iDataType;
  if (!strcmp(cDataType, "ESD")) {
    iDataType = kEsd;
  }
  else if (!strcmp(cDataType, "AOD")) {
    iDataType = kAod;
  }
  else {
    Printf("Incorrect data type option, check third argument of run macro.");
    Printf("datatype = AOD or ESD");
    return 0;
  }

  Printf("%s analysis chosen.", cDataType);

  // Return the run period to it's original capitalization.
  // It is expected to be following the form of "LHC15o".
  runPeriod = cRunPeriod;

  TString sLocalFiles(cLocalFiles);
  if (iStartAnalysis == 1) {
    if (sLocalFiles == "") {
      Printf("You need to provide the list of local files!");
      return 0;
    }
    Printf("Setting local analysis for %d files from list %s, max events = %d", iNumFiles, sLocalFiles.Data(), iNumEvents);
  }

  // Load macros needed for the analysis
  #ifndef __CLING__
  LoadMacros();
  #endif

  ////////////////////
  // Configure options
  ////////////////////
  // Select which embedding framework to use
  const bool doEmbedding = true;
  const bool fullJets = true;
  const bool enableBackgroundSubtraction = false;
  const bool embedRealData = false;
  const std::string embedRunPeriod = cEmbedRunPeriod;
  const std::string configDirBasePath = "config/embedding/";
  const bool internalEventSelection = true;
  const bool testAutoConfiguration = false;
  const bool testEmbeddingRunList = false;
  const bool testConfigureForLegoTrain = false;

  // Tagger settings
  const bool useJetTagger = true;
  const bool useResponseMaker = !useJetTagger;

  // General track and cluster cuts (used particularly for jet finding)
  const Double_t minTrackPt = 3.0;
  const Double_t minClusterPt = 3.0;

  // Define relevant variables
  const bool IsEsd = (iDataType == kEsd);
  // Note what tag we are using
  const TString tag = "new";
  // These names correspond to the _uncombined_ input objects that we are interestd in the external event
  // This allows embedding of both pythia and real data (pp)
  // We assume that pythia is the default
  TString externalEventParticlesName = "mcparticles";
  TString externalEventClustersName = "";
  if (embedRealData) {
    externalEventParticlesName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kTrack, IsEsd);
    externalEventClustersName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kCluster, IsEsd);
  }
  TString emcalCellsName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kCaloCells, IsEsd);
  TString clustersName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kCluster, IsEsd);
  TString clustersNameCombined = clustersName + "Combined";
  TString tracksName = AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kTrack, IsEsd);

  // Handle charged jets
  if (fullJets == false) {
    emcalCellsName = "";
    clustersName = "";
    clustersNameCombined = "";
  }

  ///////////////////////////////
  // Setup and Configure Analysis
  ///////////////////////////////

  // Analysis manager
  AliAnalysisManager* pMgr = new AliAnalysisManager(cTaskName);

  // Create Input Handler
  if (iDataType == kAod) {
    AliAODInputHandler * pESDHandler = AliAnalysisTaskEmcal::AddAODHandler();
  }
  else {
    AliESDInputHandler * pESDHandler = AliAnalysisTaskEmcal::AddESDHandler();
  }

  // Physics selection task
  if (iDataType == kEsd) {
    #ifdef __CLING__
    std::stringstream physSel;
    physSel << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/OADB/macros/AddTaskPhysicsSelection.C()";
    AliPhysicsSelectionTask * pPhysSelTask = reinterpret_cast<AliPhysicsSelectionTask *>(gROOT->ProcessLine(physSel.str().c_str()));
    #else
    AliPhysicsSelectionTask * pPhysSelTask = AddTaskPhysicsSelection();
    #endif
  }

  // Centrality task
  // The Run 2 condition is too restrictive, but until the switch to MultSelection is complete, it is the best we can do
  if (iDataType == kEsd && iBeamType != AliAnalysisTaskEmcal::kpp && bIsRun2 == kFALSE) {
    #ifdef __CLING__
    std::stringstream centralityTask;
    centralityTask << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/OADB/macros/AddTaskCentrality.C()";
    AliCentralitySelectionTask * pCentralityTask = reinterpret_cast<AliCentralitySelectionTask *>(gROOT->ProcessLine(centralityTask.str().c_str()));
    #else
    AliCentralitySelectionTask * pCentralityTask = AddTaskCentrality(kTRUE);
    #endif
    pCentralityTask->SelectCollisionCandidates(AliVEvent::kAny);
  }
  // AliMultSelection
  // Works for both pp and PbPb for the periods that it is calibrated
  if (bIsRun2 == kTRUE) {
    #ifdef __CLING__
    std::stringstream multSelection;
    multSelection << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C(kFALSE)";
    AliMultSelectionTask * pMultSelectionTask = reinterpret_cast<AliMultSelectionTask *>(gROOT->ProcessLine(multSelection.str().c_str()));
    #else
    AliMultSelectionTask * pMultSelectionTask = AddTaskMultSelection(kFALSE);
    #endif
    pMultSelectionTask->SelectCollisionCandidates(AliVEvent::kAny);
  }

  // CDBconnect task
  #ifdef __CLING__
  std::stringstream cdbConnect;
  cdbConnect << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/PWGPP/PilotTrain/AddTaskCDBconnect.C()";
  AliTaskCDBconnect * taskCDB = reinterpret_cast<AliTaskCDBconnect *>(gROOT->ProcessLine(cdbConnect.str().c_str()));
  #else
  AliTaskCDBconnect * taskCDB = AddTaskCDBconnect();
  #endif
  taskCDB->SetFallBackToRaw(kTRUE);

  // Debug options
  // NOTE: kDebug = 4. Can go up or down from there!
  //AliLog::SetClassDebugLevel("AliAnalysisTaskEmcalEmbeddingHelper", AliLog::kDebug+0);
  //AliLog::SetClassDebugLevel("AliYAMLConfiguration", AliLog::kDebug+0);
  //AliLog::SetClassDebugLevel("AliEmcalCorrectionTask", AliLog::kDebug+0);
  //AliLog::SetClassDebugLevel("AliEmcalCorrectionComponent", AliLog::kDebug+2);
  //AliLog::SetClassDebugLevel("AliEmcalCorrectionCellCombineCollections", AliLog::kDebug+2);
  //AliLog::SetClassDebugLevel("AliEmcalCorrectionClusterTrackMatcher", AliLog::kDebug-3);
  //AliLog::SetClassDebugLevel("AliEmcalCorrectionClusterHadronicCorrection", AliLog::kDebug-3);
  //AliLog::SetClassDebugLevel("AliEmcalJetTask", AliLog::kDebug+2);
  //AliLog::SetClassDebugLevel("AliJetContainer", AliLog::kDebug+7);
  //AliLog::SetClassDebugLevel("AliJetResponseMaker", AliLog::kDebug+0);
  //AliLog::SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations", AliLog::kDebug+2);
  AliLog::SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHPerformance", AliLog::kDebug+2);
  //AliLog::SetClassDebugLevel("AliTrackContainer", AliLog::kDebug+0);
  //AliLog::SetClassDebugLevel("AliAnalysisTask", AliLog::kDebug+7);
  //AliLog::SetClassDebugLevel("AliAnalysisTaskSE", AliLog::kDebug+7);

  if (doEmbedding) {
    // Setup embedding task
    AliAnalysisTaskEmcalEmbeddingHelper * embeddingHelper = AliAnalysisTaskEmcalEmbeddingHelper::AddTaskEmcalEmbeddingHelper();
    embeddingHelper->SelectCollisionCandidates(kPhysSel);
    //embeddingHelper->SetMaxVertexDistance(3);
    //embeddingHelper->SetZVertexCut(3);
    //embeddingHelper->SetStartingFileIndex(15);
    // Only if absolutely necessary for testing...
    //embeddingHelper->SetFilePattern("alien:///alice/sim/2012/LHC12a15e_fix/169838/%d/AOD149/");

    if (testConfigureForLegoTrain) {
      std::cout << "\tRun Macro: Using configure EMCal Embedding Helper for lego train. The following print out should have a dummy task.\n";
      embeddingHelper = AliAnalysisTaskEmcalEmbeddingHelper::ConfigureEmcalEmbeddingHelperOnLEGOTrain();
      TObjArray * tasks = pMgr->GetTasks();
      std::cout << "Tasks in Analysis Manager:\n";
      for (auto task : *tasks) {
        std::cout <<  "\t-" << task->GetName() << "\n";
      }
    }

    std::string embeddingHelperConfig = configDirBasePath + "embeddingHelper";
    embeddingHelperConfig += "_" + runPeriod;
    embeddingHelperConfig += "_" + embedRunPeriod;
    embeddingHelperConfig += ".yaml";
    embeddingHelper->SetConfigurationPath(embeddingHelperConfig.c_str());
    if (embedRealData) {
      embeddingHelper->SetTriggerMask(AliVEvent::kEMC1 | AliVEvent::kAnyINT);
      if (testEmbeddingRunList) {
        embeddingHelper->SetFilePattern("alien:///alice/data/2011/LHC11a/");
        embeddingHelper->SetInputFilename("*/ESDs/pass4_with_SDD/AOD113/*/AliAOD.root");
      }
      else {
        embeddingHelper->SetFileListFilename("aodFilesEmbedData.txt");
        // Observed to give some results with 1000 PbPb events.
        embeddingHelper->SetStartingFileIndex(2);
        embeddingHelper->SetRandomFileAccess(kFALSE);
      }
    }
    else {
      // NOTE: kAny is not equivalent to 0! Up to 24 Feb 2018, the embedding
      //       helper doesn't apply embedded event selection if equal to kAny, so kAny was
      //       equivalent to not applying any event selection. Since we intended not to
      //       apply any event selection, these gave the same outcome.
      embeddingHelper->SetTriggerMask(0);
      embeddingHelper->SetFileListFilename("aodFilesEmbed.txt");
      embeddingHelper->SetRandomFileAccess(kFALSE);
      //embeddingHelper->SetFileListFilename("tempFiles.txt");
      //embeddingHelper->SetFileListFilename("aodFilesPtHard5.txt");
      embeddingHelper->SetMCRejectOutliers();
      // Enable internal event selection
      //embeddingHelper->SetPtHardJetPtRejectionFactor(1);

      if (testAutoConfiguration)
      {
        embeddingHelper->SetAutoConfigurePtHardBins(true);
        embeddingHelper->SetAutoConfigureBasePath(".");
        embeddingHelper->SetAutoConfigureTrainTypePath("");
        embeddingHelper->SetAutoConfigureIdentifier("testEmbeddingAutoConfig");
      }
      else {
        embeddingHelper->SetPtHardBin(4);
      }
      embeddingHelper->SetNPtHardBins(11);
    }

    embeddingHelper->Initialize(testConfigureForLegoTrain);

    if (testConfigureForLegoTrain) {
      std::cout << "\tRun Macro: The dummy task should now be removed!\n";
      TObjArray * tasks = pMgr->GetTasks();
      std::cout << "Tasks in Analysis Manager:\n";
      for (auto task : *tasks) {
        std::cout <<  "\t-" << task->GetName() << "\n";
      }
    }
  }

  // EMCal corrections
  TObjArray correctionTasks;

  // Create the Correction Tasks
  correctionTasks.Add(AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("data"));
  correctionTasks.Add(AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("embed"));
  // It is important that combined is last!
  correctionTasks.Add(AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask("combined"));

  // Loop over all of the correction tasks to configure them
  AliEmcalCorrectionTask * tempCorrectionTask = 0;
  TIter next(&correctionTasks);
  while (( tempCorrectionTask = static_cast<AliEmcalCorrectionTask *>(next())))
  {
    tempCorrectionTask->SelectCollisionCandidates(kPhysSel);
    // Configure centrality
    tempCorrectionTask->SetNCentBins(5);
    if (bIsRun2) {
      tempCorrectionTask->SetUseNewCentralityEstimation(kTRUE);
    }

    std::string emcalCorrectionsConfig = configDirBasePath + "emcalCorrections";
    emcalCorrectionsConfig += "_" + runPeriod;
    emcalCorrectionsConfig += "_" + embedRunPeriod;
    emcalCorrectionsConfig += ".yaml";
    tempCorrectionTask->SetUserConfigurationFilename(emcalCorrectionsConfig);

    // The configuration file for all three is the same! They take advantage of component specialization
    if (embedRealData) {
      // Grid configuration
      //tempCorrectionTask->SetUserConfigurationFilename("alien:///alice/cern.ch/user/r/rehlersi/embedding/embedDataUserConfiguration.yaml");
    }
    else {
      // Grid configuration
      //tempCorrectionTask->SetUserConfigurationFilename("alien:///alice/cern.ch/user/r/rehlersi/embedding/embedMCUserConfiguration.yaml");
    }
    tempCorrectionTask->Initialize();
  }

  // Background
  std::string sRhoChargedName = "";
  std::string sRhoFullName = "";
  if (iBeamType != AliAnalysisTaskEmcal::kpp && enableBackgroundSubtraction == true) {
    const AliJetContainer::EJetAlgo_t rhoJetAlgorithm = AliJetContainer::kt_algorithm;
    const AliJetContainer::EJetType_t rhoJetType = AliJetContainer::kChargedJet;
    const AliJetContainer::ERecoScheme_t rhoRecoScheme = AliJetContainer::pt_scheme;
    const double rhoJetRadius = 0.4;
    sRhoChargedName = "Rho";
    sRhoFullName = "Rho_Scaled";

    AliEmcalJetTask *pKtChJetTask = AliEmcalJetTask::AddTaskEmcalJet("usedefault", "", rhoJetAlgorithm, rhoJetRadius, rhoJetType, minTrackPt, 0, kGhostArea, rhoRecoScheme, "Jet", 0., kFALSE, kFALSE);
    pKtChJetTask->SelectCollisionCandidates(kPhysSel);

    /*#ifdef __CLING__
    std::stringstream rhoTask;
    rhoTask << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C(";
    rhoTask << "\"usedefault\", ";
    rhoTask << "\"usedefault\", ";
    rhoTask << "\"" << sRhoChargedName << "\", ";
    rhoTask << 0.4 << ");";
    std::cout << "Calling rho task with " << rhoTask.str().c_str() << std::endl;
    AliAnalysisTaskRho * pRhoTask= reinterpret_cast<AliAnalysisTaskRho *>(gROOT->ProcessLine(rhoTask.str().c_str()));
    #else
    AliAnalysisTaskRho * pRhoTask = AddTaskRhoNew("usedefault", "usedefault", sRhoChargedName.c_str(), 0.4);
    #endif*/
    AliAnalysisTaskRho * pRhoTask = AddTaskRhoNew("usedefault", "usedefault", sRhoChargedName.c_str(), rhoJetRadius);
    pRhoTask->SetExcludeLeadJets(2);
    pRhoTask->SelectCollisionCandidates(kPhysSel);
    pRhoTask->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);

    TString sFuncPath = "alien:///alice/cern.ch/user/s/saiola/LHC11h_ScaleFactorFunctions.root";
    TString sFuncName = "LHC11h_HadCorr20_ClustersV2";
    pRhoTask->LoadRhoFunction(sFuncPath, sFuncName);

    // Update the track pt cut and then the jet container
    AliParticleContainer * rhoPartCont = pRhoTask->GetParticleContainer(0);
    rhoPartCont->SetParticlePtCut(3.0);
    // Must remove and recreate the jet container because the particle container used doesn't have the proper pt cut
    pRhoTask->RemoveJetContainer(0);
    pRhoTask->AddJetContainer(rhoJetType, rhoJetAlgorithm, rhoRecoScheme, rhoJetRadius, AliEmcalJet::kTPCfid, rhoPartCont, nullptr);
  }

  // Jet finding
  const AliJetContainer::EJetAlgo_t jetAlgorithm = AliJetContainer::antikt_algorithm;
  const Double_t jetRadius = 0.2;
  AliJetContainer::EJetType_t jetType = AliJetContainer::kFullJet;
  const AliJetContainer::ERecoScheme_t recoScheme = AliJetContainer::pt_scheme;
  const char * label = "Jet";
  const Double_t minJetPt = 1;
  const Bool_t lockTask = kTRUE;
  const Bool_t fillGhosts = kFALSE;

  // Do not pass clusters if we are only looking at charged jets
  if (fullJets == false) {
    jetType = AliJetContainer::kChargedJet;
    clustersName = "";
  }

  AliEmcalJetTask * pFullJetTaskPartLevelNew = 0;
  AliEmcalJetTask * pFullJetTaskDetLevelNew = 0;
  AliEmcalJetTask * pFullJetTaskNew = 0;
  AliEmcalJetTask * pFullJetTaskHybridNew = 0;
  ///////
  // Particle level PYTHIA jet finding
  ///////
  if (doEmbedding && !embedRealData) {

    pFullJetTaskPartLevelNew = AliEmcalJetTask::AddTaskEmcalJet("mcparticles", "",
            jetAlgorithm, jetRadius, jetType, minTrackPt, minClusterPt, kGhostArea, recoScheme, "partLevelJets", minJetPt, lockTask, fillGhosts);
    pFullJetTaskPartLevelNew->SelectCollisionCandidates(kPhysSel);
    pFullJetTaskPartLevelNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);

    ///////
    // External event (called embedding) settings for particle level PYTHIA jet finding
    ///////
    // Setup the tracks properly to be retrieved from the external event
    // It does not matter here if it's a Particle Container or MCParticleContainer
    AliParticleContainer * partLevelTracks = pFullJetTaskPartLevelNew->GetParticleContainer(0);
    // Called Embedded, but really just means get from an external event!
    partLevelTracks->SetIsEmbedding(kTRUE);
  }

  ///////
  // Detector level PYTHIA jet finding
  ///////
  pFullJetTaskDetLevelNew = AliEmcalJetTask::AddTaskEmcalJet(tracksName.Data(), clustersName.Data(),
          jetAlgorithm, jetRadius, jetType, minTrackPt, minClusterPt, kGhostArea, recoScheme, "detLevelJets", minJetPt, lockTask, fillGhosts);
  pFullJetTaskDetLevelNew->SelectCollisionCandidates(kPhysSel);
  pFullJetTaskDetLevelNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  pFullJetTaskDetLevelNew->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);

  ///////
  // External event (called embedding) settings for det level PYTHIA jet finding
  ///////
  // Tracks
  // Uses the name of the container passed into AliEmcalJetTask
  AliTrackContainer * tracksDetLevel = pFullJetTaskDetLevelNew->GetTrackContainer(0);
  // Get the det level tracks from the external event!
  tracksDetLevel->SetIsEmbedding(kTRUE);
  if (embedRealData) {
    tracksDetLevel->SetTrackCutsPeriod(embedRunPeriod.c_str());
  }
  // Clusters
  if (fullJets) {
    // Uses the name of the container passed into AliEmcalJetTask
    AliClusterContainer * clustersDetLevel = pFullJetTaskDetLevelNew->GetClusterContainer(0);
    // Get the det level clusters from the external event!
    clustersDetLevel->SetIsEmbedding(kTRUE);
  }

  ///////
  // PbPb jet finding
  ///////
  /*pFullJetTaskNew = AliEmcalJetTask::AddTaskEmcalJet(tracksName.Data(), clustersName.Data(),
          jetAlgorithm, jetRadius, jetType, minTrackPt, minClusterPt, kGhostArea, recoScheme, "PbPbJets", minJetPt, lockTask, fillGhosts);
  pFullJetTaskNew->SelectCollisionCandidates(kPhysSel);
  pFullJetTaskNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  if (fullJets) {
    pFullJetTaskNew->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);
  }*/

  ///////
  // PbPb + Detector level PYTHIA jet finding
  ///////
  // Sets up PbPb tracks and clusters
  // NOTE: The clusters name is different here since we output to a different branch!
  pFullJetTaskHybridNew = AliEmcalJetTask::AddTaskEmcalJet(tracksName.Data(), clustersNameCombined.Data(),
          jetAlgorithm, jetRadius, jetType, minTrackPt, minClusterPt, kGhostArea, recoScheme, "hybridLevelJets", minJetPt, lockTask, fillGhosts);
  pFullJetTaskHybridNew->SelectCollisionCandidates(kPhysSel);
  pFullJetTaskHybridNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  if (fullJets) {
    pFullJetTaskHybridNew->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);
  }
  ///////
  // External event (ie embedding) settings for PbPb jet finding (adds detector level PYTHIA)
  ///////
  // NOTE: These will break when comparing frameworks!
  // Add embedded tracks and clusters to jet finder
  // Tracks
  AliTrackContainer * tracksEmbedDetLevel = new AliTrackContainer(tracksName.Data());
  // Get the det level tracks from the external event!
  tracksEmbedDetLevel->SetIsEmbedding(kTRUE);
  tracksEmbedDetLevel->SetParticlePtCut(minTrackPt);
  if (embedRealData) {
    tracksEmbedDetLevel->SetTrackCutsPeriod(embedRunPeriod.c_str());
  }
  pFullJetTaskHybridNew->AdoptTrackContainer(tracksEmbedDetLevel);
  // Clusters
  // Already combined in clusterizer, so we shouldn't add an additional cluster container here

  //////////////////////////////////////////
  // Jet Tasks
  //////////////////////////////////////////

  // Use Sample task to assess how the embedding did
  // Need both the type and the string for various classes...
  AliEmcalJet::JetAcceptanceType acceptanceType = AliEmcalJet::kEMCALfid;
  const std::string acceptanceTypeStr = "EMCALfid";
  TString jetTag = "Jet";
  AliAnalysisTaskEmcalJetSample * sampleTaskPartLevelNew = 0;
  AliAnalysisTaskEmcalJetSample * sampleTaskDetLevelNew = 0;
  AliAnalysisTaskEmcalJetSample * sampleTaskNew = 0;
  AliAnalysisTaskEmcalJetSample * sampleTaskHybridNew = 0;
  ///////
  // Particle level PYTHIA sample task
  ///////
  if (doEmbedding && !embedRealData) {
    sampleTaskPartLevelNew = AliAnalysisTaskEmcalJetSample::AddTaskEmcalJetSample("mcparticles", "", "", "partLevelJets");

    // Set embedding
    AliParticleContainer * partCont = sampleTaskPartLevelNew->GetParticleContainer(0);
    partCont->SetIsEmbedding(kTRUE);

    partCont->SetParticlePtCut(0.15);
    sampleTaskPartLevelNew->SetHistoBins(600, 0, 300);
    sampleTaskPartLevelNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
    sampleTaskPartLevelNew->SelectCollisionCandidates(kPhysSel);

    AliJetContainer* jetCont02 = sampleTaskPartLevelNew->AddJetContainer(pFullJetTaskPartLevelNew->GetName(), acceptanceType, jetRadius);
    //AliJetContainer* jetCont02 = sampleTaskPartLevelNew->AddJetContainer(jetType, jetAlgorithm, recoScheme, jetRadius, acceptanceType, "partLevelJets");
  }

  ///////
  // Detector level PYTHIA sample task
  ///////
  /*sampleTaskDetLevelNew = AliAnalysisTaskEmcalJetSample::AddTaskEmcalJetSample(tracksName.Data(), clustersName.Data(), emcalCellsName.Data(), "detLevelJets");
  // Tracks
  // Set embedding
  AliParticleContainer * detLevelPartCont = sampleTaskDetLevelNew->GetParticleContainer(0);
  detLevelPartCont->SetIsEmbedding(kTRUE);
  // Set name to ensure no clashes
  //detLevelPartCont->SetName(detLevelPartCont->GetName() + "_Emb");
  // Settings
  detLevelPartCont->SetParticlePtCut(0.15);
  if (embedRealData) {
    dynamic_cast<AliTrackContainer *>(detLevelPartCont)->SetTrackCutsPeriod(embedRunPeriod.c_str());
  }

  if (fullJets) {
    // Clusters
    // Set embedding
    sampleTaskDetLevelNew->GetClusterContainer(0)->SetIsEmbedding(kTRUE);
    // Set name to ensure no clashes
    //sampleTaskDetLevelNew->GetClusterContainer(0)->SetName(sampleTaskDetLevelNew->GetClusterContainer(0)->GetName() + "_Emb");
    // Settings
    sampleTaskDetLevelNew->GetClusterContainer(0)->SetClusECut(0.);
    sampleTaskDetLevelNew->GetClusterContainer(0)->SetClusPtCut(0.);
    sampleTaskDetLevelNew->GetClusterContainer(0)->SetClusNonLinCorrEnergyCut(0.);
    sampleTaskDetLevelNew->GetClusterContainer(0)->SetClusHadCorrEnergyCut(0.30);
    sampleTaskDetLevelNew->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);
  }

  sampleTaskDetLevelNew->SetHistoBins(600, 0, 300);
  sampleTaskDetLevelNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  sampleTaskDetLevelNew->SelectCollisionCandidates(kPhysSel);

  AliJetContainer* detLevelJetCont02 = sampleTaskDetLevelNew->AddJetContainer(jetType, jetAlgorithm, recoScheme, jetRadius, acceptanceType, "detLevelJets");*/

  ///////
  // PbPb sample task
  ///////
  /*sampleTaskNew = AliAnalysisTaskEmcalJetSample::AddTaskEmcalJetSample(tracksName.Data(), clustersName.Data(), emcalCellsName.Data(), "PbPbJets");
  if (fullJets) {
    sampleTaskNew->GetClusterContainer(0)->SetClusECut(0.);
    sampleTaskNew->GetClusterContainer(0)->SetClusPtCut(0.);
    sampleTaskNew->GetClusterContainer(0)->SetClusNonLinCorrEnergyCut(0.);
    sampleTaskNew->GetClusterContainer(0)->SetClusHadCorrEnergyCut(0.30);
    sampleTaskNew->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);
  }
  sampleTaskNew->GetParticleContainer(0)->SetParticlePtCut(0.15);
  sampleTaskNew->SetHistoBins(600, 0, 300);
  sampleTaskNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  sampleTaskNew->SelectCollisionCandidates(kPhysSel);

  AliJetContainer* jetCont02 = sampleTaskNew->AddJetContainer(jetType, jetAlgorithm, recoScheme, jetRadius, acceptanceType, "PbPbJets");*/

  ///////
  // PbPb + Detector level PYTHIA sample task
  ///////
  // NOTE: The clusters name is different here since we output to a different branch!
  sampleTaskHybridNew = AliAnalysisTaskEmcalJetSample::AddTaskEmcalJetSample(tracksName.Data(), clustersNameCombined.Data(), "emcalCellsCombined", "hybridLevelJets");

  // PbPb tracks settings
  sampleTaskHybridNew->GetParticleContainer(0)->SetParticlePtCut(0.15);

  // Embed tracks
  tracksDetLevel = new AliTrackContainer(tracksName.Data());
  // Get the det level tracks from the external event!
  tracksDetLevel->SetIsEmbedding(kTRUE);
  // Set name to ensure no clashes
  /*TString previousName = tracksDetLevel->GetName();
  previousName += "_Emb";
  tracksDetLevel->SetName(previousName.Data());*/
  // Settings
  tracksDetLevel->SetParticlePtCut(0.15);
  if (embedRealData) {
    tracksDetLevel->SetTrackCutsPeriod(embedRunPeriod.c_str());
  }
  sampleTaskHybridNew->AdoptTrackContainer(tracksDetLevel);

  if (fullJets) {
    // PbPb + Detector Level clusters settings
    sampleTaskHybridNew->GetClusterContainer(0)->SetClusECut(0.);
    sampleTaskHybridNew->GetClusterContainer(0)->SetClusPtCut(0.);
    sampleTaskHybridNew->GetClusterContainer(0)->SetClusNonLinCorrEnergyCut(0.);
    sampleTaskHybridNew->GetClusterContainer(0)->SetClusHadCorrEnergyCut(0.30);
    sampleTaskHybridNew->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);
  }

  sampleTaskHybridNew->SetHistoBins(600, 0, 300);
  sampleTaskHybridNew->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
  sampleTaskHybridNew->SelectCollisionCandidates(kPhysSel);

  AliJetContainer * jetCont02 = sampleTaskHybridNew->AddJetContainer(pFullJetTaskHybridNew->GetName(), acceptanceType, jetRadius);
  //AliJetContainer* jetCont02 = sampleTaskHybridNew->AddJetContainer(jetType, jetAlgorithm, recoScheme, jetRadius, acceptanceType, "hybridLevelJets");

  // Jet Matching Tasks
  // 0.3 is the default max matching distance from the EMCal Jet Tagger
  double maxGeoMatchingDistance = 0.3;

  // Jet Reponse Maker
  if (useResponseMaker) {
    // Determine base (usually hybrid level) jet name
    // Can either generate it
    /*AliTrackContainer * trackCont = new AliTrackContainer(tracksName.Data());
    AliClusterContainer * clusCont = new AliClusterContainer(clustersNameCombined.Data());
    clusCont->SetClusECut(minClusterPt);
    std::string baseJetName = AliJetContainer::GenerateJetName(jetType, jetAlgorithm, recoScheme, jetRadius, trackCont, clusCont, "hybridLevelJets");*/
    // Or simply take the name from the existing jet task
    std::string baseJetName = pFullJetTaskHybridNew->GetName();
    //std::cout << "baseJetName:  " << baseJetName << "\n";

    // Determine tag (either embedded detector or embedded particle level) jet name
    // Can either generate it
    /*AliMCParticleContainer * mcPartCont = new AliMCParticleContainer(externalEventParticlesName.Data());
    std::string tagJetName = AliJetContainer::GenerateJetName(jetType, jetAlgorithm, recoScheme, jetRadius, mcPartCont, 0, "partLevelJets");*/
    // Or simply take the name from the existing jet task
    std::string tagJetName = "";
    if (embedRealData) {
      tagJetName = pFullJetTaskDetLevelNew->GetName();
    }
    else {
      tagJetName = pFullJetTaskPartLevelNew->GetName();
    }
    std::cout << "tagJetName: " << tagJetName << "\n";

    // Configure response maker
    AliJetResponseMaker::MatchingType matchingType = AliJetResponseMaker::kGeometrical;
    // Since this leading track bias applies equally to both Jet Containers and we are comparing against MC part level,
    // it is probably best to disable this and just use any bias applied at the jet finder level!
    Double_t kJetLeadingTrackBias = 0;
    // Leading hadron bias type. 0 = charged, 1 = neutral, 2 = both
    Int_t kLeadHadType = 1;
    // Type of histograms to produce. 1 = THnSparse, 0 = TH1/TH2/TH3
    Int_t kHistoType = 1;
    // Jet pt cut
    // NOTE: Setting the value here will be applied to both jet collections
    // Will set for individual jet containers below instead
    const Double_t kJetPtCut            = 0.;
    // Jet area cut
    // NOTE: Setting the value here will be applied to both jet collections
    // Will set for the det level jet container below instead (we don't want to apply to particle level)
    const Double_t kJetAreaCut          = 0;
    AliJetResponseMaker * jetResponseMatrix = AliJetResponseMaker::AddTaskJetResponseMaker(tracksName,
                  clustersNameCombined, baseJetName.c_str(), "", jetRadius,
                  externalEventParticlesName, externalEventClustersName, tagJetName.c_str(), "", jetRadius,
                  kJetPtCut, kJetAreaCut, kJetLeadingTrackBias, kLeadHadType,
                  matchingType, maxGeoMatchingDistance, maxGeoMatchingDistance, "EMCALfid", // Cut type
                  -999,-999,-999);

    // PbPb + Detector Level (Hybrid) jet
    jetResponseMatrix->GetJetContainer(0)->SetJetAcceptanceType(acceptanceType);
    // Jet-Hadron Jet area cut (det level)
    // 60% of R=0.2 jets
    jetResponseMatrix->GetJetContainer(0)->SetJetAreaCut(0.075);
    // Particle level
    // Particle level needs to come from the embedded event
    jetResponseMatrix->GetJetContainer(1)->GetParticleContainer()->SetIsEmbedding(kTRUE);
    AliClusterContainer * clusCont = jetResponseMatrix->GetJetContainer(1)->GetClusterContainer();
    // Will only exist if we are embedding real data
    if (clusCont) {
      clusCont->SetIsEmbedding(kTRUE);
    }
    jetResponseMatrix->GetJetContainer(1)->SetJetAcceptanceType(acceptanceType);
    // Remove lots of trivial jets which will pollute the matching
    jetResponseMatrix->GetJetContainer(1)->SetJetPtCut(0.15);

    // Configure task
    jetResponseMatrix->SelectCollisionCandidates(kPhysSel);
    jetResponseMatrix->SetHistoType(kHistoType);
    jetResponseMatrix->SetJetRelativeEPAngleAxis(kTRUE);
    jetResponseMatrix->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
    // Disable AliAnalysisTaskEmcal pythia functions, which won't work with the embedded event
    jetResponseMatrix->SetIsPythia(kFALSE);
  }

  if (useJetTagger) {
    /**
     * Jet taggers:
     *
     * For hybrid -> part level for a response matrix:
     * strategy is to tag:           hybrid level <-> embedded detector level
     * and:                    embedded det level <-> embedded particle level
     *
     * For hybrid -> det level for pp embedding:
     * Strategy is to only tag       hybrid level <-> embedded detector level
     * because that is sufficient for those purposes
     *
     * NOTE: The shared momentum fraction is not set here because we instead
     * handle that at the user task level to simplify configuration.
     */

    // Hybrid (PbPb + embed) jets are the "base" jet collection
    const std::string hybridLevelJetsName = pFullJetTaskHybridNew->GetName();
    // Embed det level jets are the "tag" jet collection
    const std::string detLevelJetsName = pFullJetTaskDetLevelNew->GetName();
    // Centrality estimotor
    const std::string centralityEstimator = "V0M";
    // NOTE: The AddTask macro is "AddTaskEmcalJetTaggerFast" ("Emcal" is removed for the static definition...)
    auto jetTaggerDetLevel = PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::AddTaskJetTaggerFast(
        hybridLevelJetsName.c_str(),         // "Base" jet collection which will be tagged
        detLevelJetsName.c_str(),       // "Tag" jet collection which will be used to tag (and will be tagged)
        jetRadius,                      // Jet radius
        "",                             // Hybrid ("base") rho name
        "",                             // Det level ("tag") rho name
        "",                             // tracks to attach to the jet containers. Not meaningful here, so left empty
        "",                             // clusters to attach to the jet conatiners. Not meaingful here, so left empty (plus, it's not the same for the two jet collections!)
        acceptanceTypeStr.c_str(),      // Jet acceptance type for the "base" collection
        centralityEstimator.c_str(),    // Centrality estimator
        kPhysSel,                       // Physics selection
        ""                              // Trigger class. We can just leave blank, as it's only used in the task name
        );
    // Task level settings
    jetTaggerDetLevel->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
    // Tag via geometrical matching
    jetTaggerDetLevel->SetJetTaggingMethod(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kGeo);
    // Tag the closest jet
    jetTaggerDetLevel->SetJetTaggingType(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kClosest);
    // Don't impose any additional acceptance cuts beyond the jet containers
    jetTaggerDetLevel->SetTypeAcceptance(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kNoLimit);
    // Use default matching distance
    jetTaggerDetLevel->SetMaxDistance(maxGeoMatchingDistance);
    // Redundant, but done for completeness
    jetTaggerDetLevel->SelectCollisionCandidates(kPhysSel);

    // Reapply the max track pt cut off to maintain energy resolution and avoid fake tracks
    auto hybridJetCont = jetTaggerDetLevel->GetJetContainer(0);
    hybridJetCont->SetMaxTrackPt(100);
    auto detLevelJetCont = jetTaggerDetLevel->GetJetContainer(1);
    detLevelJetCont->SetMaxTrackPt(100);

    if (!embedRealData) {
      // Embed det level jets are the "base" jet collection
      // Embed part level jets are the "tag" jet collection
      const std::string partLevelJetsName = pFullJetTaskPartLevelNew->GetName();

      auto jetTaggerPartLevel = PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::AddTaskJetTaggerFast(
        detLevelJetsName.c_str(),       // "Base" jet collection which will be tagged
        partLevelJetsName.c_str(),      // "Tag" jet collection which will be used to tag (and will be tagged)
        jetRadius,                      // Jet radius
        "",                             // Det level ("base") rho name
        "",                             // Part level ("tag") rho name
        "",                             // tracks to attach to the jet containers. Not meaningful here, so left empty
        "",                             // clusters to attach to the jet conatiners. Not meaingful here, so left empty (plus, it's not the same for the two jet collections!)
        acceptanceTypeStr.c_str(),      // Jet acceptance type for the "base" collection
        centralityEstimator.c_str(),    // Centrality estimator
        kPhysSel,                       // Physics selection
        ""                              // Trigger class. We can just leave blank, as it's only used in the task name
        );
      // Task level settings
      jetTaggerPartLevel->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);
      // Tag via geometrical matching
      jetTaggerPartLevel->SetJetTaggingMethod(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kGeo);
      // Tag the closest jet
      jetTaggerPartLevel->SetJetTaggingType(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kClosest);
      // Don't impose any additional acceptance cuts beyond the jet containers
      jetTaggerPartLevel->SetTypeAcceptance(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTaskFast::kNoLimit);
      // Use default matching distance
      jetTaggerPartLevel->SetMaxDistance(maxGeoMatchingDistance);
      // Redundant, but done for completeness
      jetTaggerPartLevel->SelectCollisionCandidates(kPhysSel);

      // Reapply the max track pt cut off to maintain energy resolution and avoid fake tracks
      // However, don't apply to the particle level jets which don't suffer this effect
      detLevelJetCont = jetTaggerPartLevel->GetJetContainer(0);
      detLevelJetCont->SetMaxTrackPt(100);
    }
  }

  //////////////////////
  // Jet-H analysis task
  //////////////////////
  const std::string jetHClustersName = embedRealData ? externalEventClustersName.Data() : clustersNameCombined.Data();
  const std::string jetHTracksName = embedRealData ? externalEventParticlesName.Data() : AliEmcalContainerUtils::DetermineUseDefaultName(AliEmcalContainerUtils::kTrack, IsEsd);
  const Double_t trackBias = PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations::kDisableBias;
  // NOTE: Careful here with the cluster bias! Set low to ensure that there are entries in pp!
  const Double_t clusterBias = 6.0;
  const Int_t nTracksMixedEvent = 50000;
  const Int_t minNTracksMixedEvent = 5000;
  const Int_t minNEventsMixedEvent = 1;
  const UInt_t nCentBinsMixedEvent = 10;
  UInt_t triggerEventsSelection = 0;
  UInt_t mixedEventsSelection = 0;
  // Need to set embedded event selection!
  if (iBeamType == AliAnalysisTaskEmcal::kpp || (doEmbedding && embedRealData)) {
    std::cout << "Selecting pp trigger selections for the jetH task.\n";
    // NOTE: kINT1 == kMB! Thus, kINT1 is implicitly included in kAnyINT!
    triggerEventsSelection = AliVEvent::kEMC1 | AliVEvent::kAnyINT;
    mixedEventsSelection = AliVEvent::kAnyINT;
  }
  else {
    if (doEmbedding) {
      std::cout << "Selecting event selection of \"0\" for jetH task triggers and mixed event selection due to embedding MC.\n";
      triggerEventsSelection = 0;
      mixedEventsSelection = triggerEventsSelection;
    }
    else {
      std::cout << "Selecting PbPb trigger selections for the jetH task.\n";
      // NOTE: These still may not be the proper conditions
      // We want to include everything that is embedded in a reasonable event!
      triggerEventsSelection = AliVEvent::kEMCEGA | AliVEvent::kAnyINT | AliVEvent::kCentral | AliVEvent::kSemiCentral;
      mixedEventsSelection = triggerEventsSelection;
    }
  }
  const Bool_t sparseAxes = kTRUE;
  const Bool_t widerTrackPtBins = kTRUE;
  // Efficiency correction
  auto efficiencyCorrectionType = PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations::kEffAutomaticConfiguration;
  if (doEmbedding) {
    // Explicitly configured to use pp efficiency since we are embedding
    efficiencyCorrectionType = PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations::kEffPP;
  }
  // JES correction
  const Bool_t embeddingCorrection = kFALSE;
  // Local tests
  const char * embeddingCorrectionFilename = "../embeddingCorrection.root";
  const char * embeddingCorrectionHistName = "embeddingCorrection";

  std::cout << "externalEventParticlesName: " << externalEventParticlesName << "\n";
  std::cout << "JetH tracks name: " << jetHTracksName << "\n";
  std::cout << "externalEventClustersName: " << externalEventClustersName << "\n";
  std::cout << "JetH clusters name: " << jetHClustersName << "\n";
  //auto jetH = PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations::AddTaskEmcalJetHCorrelations(jetHTracksName.c_str(), jetHClustersName.c_str(),
  auto jetH = AddTaskEmcalJetHCorrelations(jetHTracksName.c_str(), jetHClustersName.c_str(),
      trackBias, clusterBias,                                     // Track, cluster bias
      nTracksMixedEvent, minNTracksMixedEvent, minNEventsMixedEvent, nCentBinsMixedEvent, // Mixed event options
      triggerEventsSelection,                                     // Trigger events
      mixedEventsSelection,                                       // Mixed event
      sparseAxes, widerTrackPtBins,                               // Less sprase axis, wider binning
      efficiencyCorrectionType, embeddingCorrection,              // Track efficiency, embedding correction
      embeddingCorrectionFilename, embeddingCorrectionHistName,   // Settings for embedding
      "new"                                                       // Suffix
      );

  if (doEmbedding) {
    jetH->ConfigureForEmbeddingAnalysis();
    jetH->SetMaximumMatchedJetDistance(maxGeoMatchingDistance);
    // Test options
    if (!embedRealData) {
      jetH->SetRequireMatchedPartLevelJet(true);
    }
  }
  else {
    jetH->ConfigureForStandardAnalysis();
  }
  jetH->SelectCollisionCandidates(kPhysSel);
  std::string jetHConfig = configDirBasePath + "jetHCorrelations";
  jetHConfig += "_" + runPeriod;
  jetHConfig += ".yaml";
  jetH->AddConfigurationFile(jetHConfig, "config");
  jetH->Initialize();
  // Override, just in case.
  jetH->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);

  // Test artificial track inefficiency
  //jetH->SetArtificialTrackingInefficiency(.3);
  // Test minimum shared momentum fraction
  //jetH->SetMinimumSharedMomentumFraction(.5);

  // Jet-h performance task
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHPerformance * jetHPerformance = AddTaskEmcalJetHPerformance();
  jetHPerformance->SelectCollisionCandidates(kPhysSel);
  // Setup configuration
  // Base configuration that will be overridden.
  std::string jetHPerformanceConfigBase = configDirBasePath + "jetHPerformance";
  jetHPerformanceConfigBase += "_" + runPeriod;
  // Period specific configuration.
  std::string jetHPerformanceConfigPeriodSpecific = jetHPerformanceConfigBase;
  // Finalize base config name and set it.
  jetHPerformanceConfigBase += "_base";
  jetHPerformanceConfigBase += ".yaml";
  jetHPerformance->AddConfigurationFile(jetHPerformanceConfigBase, "base");
  // Finalize period specific config name and set it.
  jetHPerformanceConfigPeriodSpecific += "_" + embedRunPeriod;
  jetHPerformanceConfigPeriodSpecific += ".yaml";
  jetHPerformance->AddConfigurationFile(jetHPerformanceConfigPeriodSpecific, "periodSpecific");
  // Complete setup.
  jetHPerformance->Initialize();
  // Override, just in case.
  jetHPerformance->SetRecycleUnusedEmbeddedEventsMode(internalEventSelection);

  TObjArray *pTopTasks = pMgr->GetTasks();
  for (Int_t i = 0; i < pTopTasks->GetEntries(); ++i) {
    AliAnalysisTaskSE *pTask = dynamic_cast<AliAnalysisTaskSE*>(pTopTasks->At(i));
    if (!pTask) continue;
    if (pTask->InheritsFrom("AliAnalysisTaskEmcal")) {
      AliAnalysisTaskEmcal *pTaskEmcal = static_cast<AliAnalysisTaskEmcal*>(pTask);
      Printf("Setting beam type %d for task %s", iBeamType, pTaskEmcal->GetName());
      pTaskEmcal->SetForceBeamType(iBeamType);
    }
    if (pTask->InheritsFrom("AliEmcalCorrectionTask")) {
      AliEmcalCorrectionTask * pTaskEmcalCorrection = static_cast<AliEmcalCorrectionTask*>(pTask);
      Printf("Setting beam type %d for task %s", iBeamType, pTaskEmcalCorrection->GetName());
      pTaskEmcalCorrection->SetForceBeamType(static_cast<AliEmcalCorrectionTask::BeamType>(iBeamType));
    }
  }

  if (!pMgr->InitAnalysis()) return 0;
  pMgr->PrintStatus();

  //pMgr->SetDebugLevel(10);
  pMgr->SetUseProgressBar(kTRUE, 250);

  TFile *pOutFile = new TFile("train.root","RECREATE");
  pOutFile->cd();
  pMgr->Write();
  pOutFile->Close();
  delete pOutFile;

  if (iStartAnalysis == 1) { // start local analysis
    TChain* pChain = 0;
    if (iDataType == kAod) {
      #ifdef __CLING__
      std::stringstream aodChain;
      aodChain << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/PWG/EMCAL/macros/CreateAODChain.C(";
      aodChain << "\"" << sLocalFiles.Data() << "\", ";
      aodChain << iNumEvents << ", ";
      aodChain << 0 << ", ";
      aodChain << std::boolalpha << kFALSE << ");";
      pChain = reinterpret_cast<TChain *>(gROOT->ProcessLine(aodChain.str().c_str()));
      #else
      gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/CreateAODChain.C");
      pChain = CreateAODChain(sLocalFiles.Data(), iNumFiles, 0, kFALSE);
      #endif
    }
    else {
      #ifdef __CLING__
      std::stringstream esdChain;
      esdChain << ".x " << gSystem->Getenv("ALICE_PHYSICS") <<  "/PWG/EMCAL/macros/CreateESDChain.C(";
      esdChain << "\"" << sLocalFiles.Data() << "\", ";
      esdChain << iNumEvents << ", ";
      esdChain << 0 << ", ";
      esdChain << std::boolalpha << kFALSE << ");";
      pChain = reinterpret_cast<TChain *>(gROOT->ProcessLine(esdChain.str().c_str()));
      #else
      gROOT->LoadMacro("$ALICE_PHYSICS/PWG/EMCAL/macros/CreateESDChain.C");
      pChain = CreateESDChain(sLocalFiles.Data(), iNumFiles, 0, kFALSE);
      #endif
    }

    // start analysis
    Printf("Starting Analysis...");
    pMgr->StartAnalysis("local", pChain, iNumEvents);
  }
  else if (iStartAnalysis == 2) {  // start grid analysis
    StartGridAnalysis(pMgr, cTaskName, cGridMode);
  }

  return pMgr;
}

void LoadMacros()
{
  // Aliroot macros
  gROOT->LoadMacro("$ALICE_PHYSICS/OADB/macros/AddTaskCentrality.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/PWGPP/PilotTrain/AddTaskCDBconnect.C");
  gROOT->LoadMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C");
}

void StartGridAnalysis(AliAnalysisManager* pMgr, const char* uniqueName, const char* cGridMode)
{
  Int_t maxFilesPerWorker = 4;
  Int_t workerTTL = 7200;
  const char* runNumbers = "180720";
  const char* pattern = "pass2/AOD/*/AliAOD.root";
  const char* gridDir = "/alice/data/2012/LHC12c";
  const char* additionalCXXs = "";
  const char* additionalHs = "";

  AliAnalysisGrid *plugin = CreateAlienHandler(uniqueName, gridDir, cGridMode, runNumbers, pattern, additionalCXXs, additionalHs, maxFilesPerWorker, workerTTL, kFALSE);
  pMgr->SetGridHandler(plugin);

  // start analysis
  Printf("Starting GRID Analysis...");
  pMgr->SetDebugLevel(0);
  pMgr->StartAnalysis("grid");
}

AliAnalysisGrid* CreateAlienHandler(const char* uniqueName, const char* gridDir, const char* gridMode, const char* runNumbers,
    const char* pattern, TString additionalCode, TString additionalHeaders, Int_t maxFilesPerWorker, Int_t workerTTL, Bool_t isMC)
{
  TDatime currentTime;
  TString tmpName(uniqueName);

  // Only add current date and time when not in terminate mode! In this case the exact name has to be supplied by the user
  if (strcmp(gridMode, "terminate")) {
    tmpName += "_";
    tmpName += currentTime.GetDate();
    tmpName += "_";
    tmpName += currentTime.GetTime();
  }

  TString macroName("");
  TString execName("");
  TString jdlName("");
  macroName = Form("%s.C", tmpName.Data());
  execName = Form("%s.sh", tmpName.Data());
  jdlName = Form("%s.jdl", tmpName.Data());

  AliAnalysisAlien *plugin = new AliAnalysisAlien();
  plugin->SetOverwriteMode();
  plugin->SetRunMode(gridMode);

  // Here you can set the (Ali)PHYSICS version you want to use
  plugin->SetAliPhysicsVersion("vAN-20160203-1");

  plugin->SetGridDataDir(gridDir); // e.g. "/alice/sim/LHC10a6"
  plugin->SetDataPattern(pattern); //dir structure in run directory

  if (!isMC) plugin->SetRunPrefix("000");

  plugin->AddRunList(runNumbers);

  plugin->SetGridWorkingDir(Form("work/%s",tmpName.Data()));
  plugin->SetGridOutputDir("output"); // In this case will be $HOME/work/output

  plugin->SetAnalysisSource(additionalCode.Data());

  plugin->SetDefaultOutputs(kTRUE);
  plugin->SetAnalysisMacro(macroName.Data());
  plugin->SetSplitMaxInputFileNumber(maxFilesPerWorker);
  plugin->SetExecutable(execName.Data());
  plugin->SetTTL(workerTTL);
  plugin->SetInputFormat("xml-single");
  plugin->SetJDLName(jdlName.Data());
  plugin->SetPrice(1);
  plugin->SetSplitMode("se");

  // merging via jdl
  plugin->SetMergeViaJDL(kTRUE);
  plugin->SetOneStageMerging(kFALSE);
  plugin->SetMaxMergeStages(2);

  return plugin;
}
