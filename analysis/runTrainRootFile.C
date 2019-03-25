// Tests train.root files genreated by AliAnalysisManager
// Derived from runEMCalCorrections.C

class AliAnalysisManager;

void runTrainRootFile(
        const char   *filePath       = "train.root",            // Location of the train.root file
        const char   *cDataType      = "AOD",                   // set the analysis type, AOD or ESD
        const char   *cLocalFiles    = "LHC11h.txt",          // set the local list file
        UInt_t        iNumFiles      = 100,                     // number of files analyzed locally
        UInt_t        iNumEvents     = 1000                     // number of events to be analyzed
    )
{
  // Setup the AliAnalysisManager
  TFile * fIn = TFile::Open(filePath);
  auto keys = fIn->GetListOfKeys();
  if (keys->GetEntries() > 1) {
    ::Error("runTrainRootFile", "Cannot determine analysis manager due to too many keys in the file. Check that only the analysis manager in stored in the train.root file!");
  }

  // Get the analysis manager
  AliAnalysisManager * pMgr = 0;
  TKey * key = 0;
  for (int i = 0; i < keys->GetEntries(); i++)
  {
    key = dynamic_cast<TKey *>(keys->At(i));
    pMgr = dynamic_cast<AliAnalysisManager *>(key->ReadObj());
  }

  TString sLocalFiles(cLocalFiles);
  if (sLocalFiles == "") {
    ::Error("runTrainRootFile", "You need to provide the list of local files!");
    return 0;
  }
  Printf("Setting local analysis for %d files from list %s, max events = %d", iNumFiles, sLocalFiles.Data(), iNumEvents);

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

  // Set debug levels here
  //AliLog::SetClassDebugLevel("AliAnalysisTaskEmcalEmbeddingHelper", AliLog::kDebug-2);
  AliLog::SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHPerformance", AliLog::kDebug+0);

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

  // Print status and control the progress information
  pMgr->PrintStatus();
  pMgr->SetUseProgressBar(kTRUE, 250);

  // start analysis
  Printf("Starting Analysis...");
  pMgr->StartAnalysis("local", pChain, iNumEvents);
}
