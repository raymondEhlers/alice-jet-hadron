/// \file runJetHAnalysis.C
/// \brief Jet-Hadron away-side analysis
///
/// \ingroup EMCALJETFW
/// Run macro for the Jet-Hadron away-side analysis
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

class AliEmcalCorrectionTask;
class AliEmcalJetTask;
class AliAnalysisTaskEmcalJetSample;
namespace PWGJE {
  namespace EMCALJetTasks {
    class AliAnalysisTaskEmcalJetHPerformance;
    class AliAnalysisTaskEmcalJetHCorrelations;
  }
}

#ifdef __CLING__
// Tell ROOT where to find AliRoot headers
R__ADD_INCLUDE_PATH($ALICE_ROOT)
// Tell ROOT where to find AliPhysics headers
R__ADD_INCLUDE_PATH($ALICE_PHYSICS)
// Common tasks
#include "OADB/macros/AddTaskPhysicsSelection.C"
#include "OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C"
#include "OADB/macros/AddTaskCentrality.C"
#include "PWGPP/PilotTrain/AddTaskCDBconnect.C"
// Simplify rho task usage
#include "PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C"
// Include AddTask to test for the LEGO train
#include "PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetHCorrelations.C"
#include "PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetHPerformance.C"
#endif

void LoadMacros();
void StartGridAnalysis(AliAnalysisManager* pMgr, const char* uniqueName, const char* cGridMode);
AliAnalysisGrid* CreateAlienHandler(const char* uniqueName, const char* gridDir, const char* gridMode, const char* runNumbers,
    const char* pattern, TString additionalCode, TString additionalHeaders, Int_t maxFilesPerWorker, Int_t workerTTL, Bool_t isMC);

//______________________________________________________________________________
AliAnalysisManager* runJetHAnalysis(
    const char   *cDataType      = "AOD",                                   // set the analysis type, AOD or ESD
    const char   *cRunPeriod     = "LHC15o",                                // set the run period
    const char   *cLocalFiles    = "aodFiles.txt",                          // set the local list file
    const UInt_t  iNumEvents     = 1000,                                    // number of events to be analyzed
    const UInt_t  kPhysSel       = AliVEvent::kAnyINT,
                   //AliVEvent::kEMC1 | AliVEvent::kAnyINT,
                   //AliVEvent::kEMCEGA | AliVEvent::kAnyINT |
                   //AliVEvent::kCentral | AliVEvent::kSemiCentral, //AliVEvent::kAny,                         // physics selection
    const char   *cTaskName      = "EMCalJetHAnalysis",                     // sets name of analysis manager
    // 0 = only prepare the analysis manager but do not start the analysis
    // 1 = prepare the analysis manager and start the analysis
    // 2 = launch a grid analysis
    Int_t         iStartAnalysis = 1,
    const UInt_t  iNumFiles      = 5,                                     // number of files analyzed locally
    const char   *cGridMode      = "test"
)
{
  // Setup period
  TString sRunPeriod(cRunPeriod);
  sRunPeriod.ToLower();

  // Set Run 2
  Bool_t bIsRun2 = kFALSE;
  if (sRunPeriod.Length() == 6 && (sRunPeriod.BeginsWith("lhc15") || sRunPeriod.BeginsWith("lhc16") || sRunPeriod.BeginsWith("lhc17"))) bIsRun2 = kTRUE;

  // Set beam type
  AliAnalysisTaskEmcal::BeamType iBeamType = AliAnalysisTaskEmcal::kpp;
  if (sRunPeriod == "lhc10h" || sRunPeriod == "lhc11h" || sRunPeriod == "lhc15o") {
    iBeamType = AliAnalysisTaskEmcal::kAA;
  }
  else if (sRunPeriod == "lhc12g" || sRunPeriod == "lhc13b" || sRunPeriod == "lhc13c" ||
      sRunPeriod == "lhc13d" || sRunPeriod == "lhc13e" || sRunPeriod == "lhc13f" ||
      sRunPeriod == "lhc16q" || sRunPeriod == "lhc16r" || sRunPeriod == "lhc16s" || sRunPeriod == "lhc16t") {
    iBeamType = AliAnalysisTaskEmcal::kpA;
  }

  // Ghost area
  Double_t kGhostArea = 0.01;
  if (iBeamType != AliAnalysisTaskEmcal::kpp) kGhostArea = 0.005;

  // Setup track container
  AliTrackContainer::SetDefTrackCutsPeriod(sRunPeriod);
  Printf("Default track cut period set to: %s", AliTrackContainer::GetDefTrackCutsPeriod().Data());

  // Configuration
  // General track and cluster cuts (used particularly for jet finding)
  const Double_t minTrackPt = 3.0;
  const Double_t minClusterPt = 3.0;

  // Control background subtraction
  Bool_t bEnableBackgroundSubtraction = kFALSE;

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

  // Analysis manager
  AliAnalysisManager* pMgr = new AliAnalysisManager(cTaskName);

  if (iDataType == kAod) {
    AliAODInputHandler * pESDHandler = AliAnalysisTaskEmcal::AddAODHandler();
  }
  else {  
    AliESDInputHandler * pESDHandler = AliAnalysisTaskEmcal::AddESDHandler();
  }

  // CDBconnect task
  AliTaskCDBconnect * taskCDB = AddTaskCDBconnect();
  taskCDB->SetFallBackToRaw(kTRUE);

  // Physics selection task
  if (iDataType == kEsd) {
    AliPhysicsSelectionTask * pPhysSelTask = AddTaskPhysicsSelection();
  }

  // Centrality task
  // The Run 2 condition is too restrictive, but until the switch to MultSelection is complete, it is the best we can do
  if (iDataType == kEsd && iBeamType != AliAnalysisTaskEmcal::kpp && bIsRun2 == kFALSE) {
    AliCentralitySelectionTask * pCentralityTask = AddTaskCentrality(kTRUE);
    pCentralityTask->SelectCollisionCandidates(AliVEvent::kAny);
  }
  // AliMultSelection
  // Works for both pp and PbPb for the periods that it is calibrated
  if (bIsRun2 == kTRUE) {
    AliMultSelectionTask * pMultSelectionTask = AddTaskMultSelection(kFALSE);
    pMultSelectionTask->SelectCollisionCandidates(AliVEvent::kAny);
  }

  /////////////////
  // Debug settings
  /////////////////
  //AliLog::SetClassDebugLevel("AliEmcalCorrectionComponent", AliLog::kDebug+3);
  //AliLog::SetClassDebugLevel("AliAnalysisTaskEmcalJetHCorrelations", AliLog::kDebug+1);
  AliLog::SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHPerformance", AliLog::kDebug+5);
  //AliLog::SetClassDebugLevel("AliJetContainer", AliLog::kDebug+7);

  // EMCal corrections
  AliEmcalCorrectionTask * correctionTask = AliEmcalCorrectionTask::AddTaskEmcalCorrectionTask();
  correctionTask->SelectCollisionCandidates(kPhysSel);
  correctionTask->SetNCentBins(5);
  correctionTask->SetUseNewCentralityEstimation(bIsRun2);
  // Local configuration
  std::string emcalCorrectionsConfig = "config/emcalCorrectionsConfig";
  emcalCorrectionsConfig += cRunPeriod;
  emcalCorrectionsConfig += ".yaml";
  correctionTask->SetUserConfigurationFilename(emcalCorrectionsConfig);
  // Grid configuration
  //correctionTask->SetUserConfigurationFilename("alien:///alice/cern.ch/user/r/rehlersi/jetH/jetHUserConfiguration.yaml");
  correctionTask->Initialize();

  // Background
  std::string sRhoChargedName = "";
  std::string sRhoFullName = "";
  if (iBeamType != AliAnalysisTaskEmcal::kpp && bEnableBackgroundSubtraction == kTRUE) {
    const AliJetContainer::EJetAlgo_t rhoJetAlgorithm = AliJetContainer::kt_algorithm;
    const AliJetContainer::EJetType_t rhoJetType = AliJetContainer::kChargedJet;
    const AliJetContainer::ERecoScheme_t rhoRecoScheme = AliJetContainer::pt_scheme;
    const double rhoJetRadius = 0.4;
    sRhoChargedName = "Rho";
    sRhoFullName = "Rho_Scaled";

    AliEmcalJetTask *pKtChJetTask = AliEmcalJetTask::AddTaskEmcalJet("usedefault", "", rhoJetAlgorithm, rhoJetRadius, rhoJetType, minTrackPt, 0, kGhostArea, rhoRecoScheme, "Jet", 0., kFALSE, kFALSE);
    pKtChJetTask->SelectCollisionCandidates(kPhysSel);
    pKtChJetTask->SetUseNewCentralityEstimation(bIsRun2);
    pKtChJetTask->SetNCentBins(5);

    AliAnalysisTaskRho * pRhoTask = AddTaskRhoNew("usedefault", "usedefault", sRhoChargedName.c_str(), rhoJetRadius);
    pRhoTask->SetExcludeLeadJets(2);
    pRhoTask->SelectCollisionCandidates(kPhysSel);

    TString sFuncPath = "alien:///alice/cern.ch/user/s/saiola/LHC11h_ScaleFactorFunctions.root";
    TString sFuncName = "LHC11h_HadCorr20_ClustersV2";
    pRhoTask->LoadRhoFunction(sFuncPath, sFuncName);
  }

  // Jet finding
  const AliJetContainer::EJetAlgo_t jetAlgorithm = AliJetContainer::antikt_algorithm;
  const AliJetContainer::EJetType_t jetType = AliJetContainer::kFullJet;
  const AliJetContainer::ERecoScheme_t recoScheme = AliJetContainer::pt_scheme;
  const double jetRadius = 0.2;
  const char * label = "Jet";
  const double minJetPt = 1;
  const bool lockTask = kTRUE;
  const bool fillGhosts = kFALSE;

  AliEmcalJetTask *pFullJet02TaskNew = AliEmcalJetTask::AddTaskEmcalJet("usedefault", "usedefault",
          jetAlgorithm, jetRadius, jetType, minTrackPt, minClusterPt, kGhostArea, recoScheme, label, minJetPt, lockTask, fillGhosts);
  pFullJet02TaskNew->SelectCollisionCandidates(kPhysSel);
  pFullJet02TaskNew->SetUseNewCentralityEstimation(bIsRun2);
  pFullJet02TaskNew->SetNCentBins(5);
  pFullJet02TaskNew->GetClusterContainer(0)->SetDefaultClusterEnergy(AliVCluster::kHadCorr);

  //////////////////////////////////////////
  // Jet-H Task
  //////////////////////////////////////////

  const double trackBias = PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations::kDisableBias;
  // NOTE: Careful here with the cluster bias! Set low to ensure that there are entries in pp!
  const double clusterBias = 6.0;
  const Int_t nTracksMixedEvent = 50000;
  const Int_t minNTracksMixedEvent = 5000;
  const Int_t minNEventsMixedEvent = 1;
  const UInt_t nCentBinsMixedEvent = 10;
  UInt_t triggerEventsSelection = 0;
  UInt_t mixedEventsSelection = 0;
  if (iBeamType == AliAnalysisTaskEmcal::kpp) {
    // NOTE: kINT1 == kMB! Thus, kINT1 is implicitly included in kAnyINT!
    triggerEventsSelection = AliVEvent::kEMC1 | AliVEvent::kAnyINT;
    mixedEventsSelection = AliVEvent::kAnyINT;
  }
  else {
    if (sRunPeriod == "lhc11h") {
      triggerEventsSelection = AliVEvent::kEMCEGA;
      mixedEventsSelection = AliVEvent::kAnyINT | AliVEvent::kCentral | AliVEvent::kSemiCentral;
    }
    else {
      triggerEventsSelection = AliVEvent::kAnyINT;
      mixedEventsSelection = AliVEvent::kAnyINT;
    }
  }
  const bool sparseAxes = true;
  const bool widerTrackPtBins = true;
  const auto efficiencyCorrectionType = PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations::kEffAutomaticConfiguration;
  const bool embeddingCorrection = false;
  // Local tests
  const char * embeddingCorrectionFilename = "../embeddingCorrection.root";
  const char * embeddingCorrectionHistName = "embeddingCorrection";
  // Grid tests
  /*const char * embeddingCorrectionFilename = "alien:///alice/cern.ch/user/r/rehlersi/jetH/pp/embeddingCorrection.TH2Ds.root";
  const char * embeddingCorrectionHistName = "embeddingCorrection_Clus6";*/
  //const char * suffix = "clusbias5R2GANew";

  // Connect to AliEn if necessary
  /*TString embeddingCorrectionFilenameStr = embeddingCorrectionFilename;
  if (embeddingCorrectionFilenameStr.Contains("alien://") && !gGrid) {
    TGrid::Connect("alien://");
  }*/

  //auto jetHTask = PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHCorrelations::AddTaskEmcalJetHCorrelations("usedefault", "usedefault",
  auto jetHTask = AddTaskEmcalJetHCorrelations("usedefault", "usedefault",
      trackBias, clusterBias,                                     // Track, cluster bias
      nTracksMixedEvent, minNTracksMixedEvent, minNEventsMixedEvent, nCentBinsMixedEvent, // Mixed event options
      triggerEventsSelection,                                     // Trigger events
      mixedEventsSelection,                                       // Mixed event
      sparseAxes, widerTrackPtBins,                               // Less sprase axis, wider binning
      efficiencyCorrectionType, embeddingCorrection,              // Track efficiency, embedding correction
      embeddingCorrectionFilename, embeddingCorrectionHistName,   // Settings for embedding
      "jetH"                                                      // Suffix
      );

  // Setup JES correction
  //jetHTaskNewFramework->RetrieveAndInitializeJESCorrectionHist(embeddingCorrectionFilename, embeddingCorrectionHistName, AliAnalysisTaskEmcalJetHCorrelations::kDisableBias, 6);

  // The combination of the two are the ones we are interested in
  jetHTask->SelectCollisionCandidates(triggerEventsSelection | mixedEventsSelection);

  // Disable task for fast partition trigger
  //jetHTask->SetDisableFastPartition(kTRUE);

  // Configure the task
  jetHTask->SetUseNewCentralityEstimation(bIsRun2);
  jetHTask->ConfigureForStandardAnalysis();

  if (iBeamType != AliAnalysisTaskEmcal::kpp && bEnableBackgroundSubtraction == kTRUE) {
    //AliJetContainer* jetCont = jetHTask->AddJetContainer(AliJetContainer::kFullJet, AliJetContainer::antikt_algorithm, AliJetContainer::pt_scheme, 0.2, AliEmcalJet::kEMCALfid, "Jet");
    AliJetContainer * jetCont = jetHTask->GetJetContainer(0);
    jetCont->SetRhoName(sRhoFullName.c_str());
    jetCont->SetPercAreaCut(0.6);
  }

  // Complete configuration.
  std::string jetHCorrelationsConfigurationName = "config/jetHCorrelations.yaml";
  std::cout << "Using jet-h correlations configuration " << jetHCorrelationsConfigurationName << "\n";
  jetHTask->AddConfigurationFile(jetHCorrelationsConfigurationName, "config");
  jetHTask->Initialize();

  //////////////////////////////////////////
  // Jet-H performance task for QA-like information
  //////////////////////////////////////////
  PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHPerformance * jetHPerformance = AddTaskEmcalJetHPerformance();
  jetHPerformance->SelectCollisionCandidates(kPhysSel);
  std::string jetHPerformanceConfigurationName = "config/jetHPerformance.yaml";
  std::cout << "Using jet-h performance configuration " << jetHPerformanceConfigurationName << "\n";
  jetHPerformance->SetUseNewCentralityEstimation(bIsRun2);
  jetHPerformance->AddConfigurationFile(jetHPerformanceConfigurationName, "config");
  jetHPerformance->Initialize();

  /*
  AliAnalysisTaskEmcalJetSample * sampleTaskNew = AliAnalysisTaskEmcalJetSample::AddTaskEmcalJetSample("usedefault", "usedefault", "usedefault");
  sampleTaskNew->GetClusterContainer(0)->SetClusECut(3.);
  sampleTaskNew->GetParticleContainer(0)->SetParticlePtCut(3);
  sampleTaskNew->SetHistoBins(600, 0, 300);
  sampleTaskNew->SelectCollisionCandidates(kPhysSel);

  AliJetContainer* jetCont02 = sampleTaskNew->AddJetContainer(jetType, jetAlgorithm, recoScheme, jetRadius, AliEmcalJet::kEMCALfid);
  */

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
  // LHC11a
  // /alice/data/2011/LHC11a/000146860/ESDs/pass4_with_SDD/AOD113/0002
  // Run list from EMC_pp
  const char* runNumbers = "146860 146859 146858 146856 146824 146817 146807 146806 146805 146804 146803 146802 146801 146748 146747 146746";
  const char* pattern = "ESDs/pass4_with_SDD/AOD113/*/AliAOD.root";
  const char* gridDir = "/alice/data/2011/LHC11a";
  // LHC11h
  // /alice/data/2011/LHC11h_2/000167693/ESDs/pass2/AOD145
  //const char* runNumbers = "167693";
  //const char* pattern = "ESDs/pass2/AOD145/*/AliAOD.root";
  //const char* gridDir = "/alice/data/2011/LHC11h_2";
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
  plugin->SetAliPhysicsVersion("vAN-20170103-1");

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
