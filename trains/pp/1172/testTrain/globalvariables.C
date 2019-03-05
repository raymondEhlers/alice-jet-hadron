{
// Set this variable to the least inclusive trigger set that includes all user requested triggers (0 means no selection applied)
//UInt_t kComPhysSel = AliVEvent::kAnyINT;
//UInt_t kComPhysSel = AliVEvent::kAny;
UInt_t kComPhysSel = 0;

// No need to change any of the following trigger sets, but can add more variables as requested by users
UInt_t kPrePhysSel = AliVEvent:: kAny;
UInt_t kPhysSelMB = AliVEvent::kMB;
UInt_t kPhysSel_DmesonJet = 1<<31; //AliEmcalPhysicsSelection::kEmcalOk
UInt_t kPhysSelINT7 = AliVEvent::kINT7;

// Name of the collections

//TString kTracksName = "PicoTracks";
TString kTracksName2 = "PicoTracks2";
TString kInTracksName = "AODFilterTracks";
TString kTracksName = "tracks";

//Chiara
const char* kInputTracks = "tracks";
const char* kUsedTracks = "PicoTracks";
Bool_t kIsPythia=kFALSE;
Bool_t isAOD = 1;

TString kClusName = "caloClusters";
TString kEmcalTracksName = "EmcalTracks";
TString kEmcalClusName = "EmcalClusters";
TString kCorrClusName = "CaloClustersCorr";
TString kMCTracksName = "mcparticles";
TString kEmcalCellsName = "emcalCells";


Bool_t kMakeEmcalTriggers = kTRUE;
TString kEmcalTriggers = "";
if (kMakeEmcalTriggers) kEmcalTriggers = "EmcalTriggers";

Bool_t kDoAODTrackProp=kFALSE;

TString kTpcKtJetsName(Form("Jet_KTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR02Name(Form("Jet_AKTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR03Name(Form("Jet_AKTChargedR030_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR04Name(Form("Jet_AKTChargedR040_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR06Name(Form("Jet_AKTChargedR060_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR02EschemeName(Form("Jet_AKTChargedR020_%s_pT0150_E_scheme",kTracksName.Data()));
TString kTpcJetsR03EschemeName(Form("Jet_AKTChargedR030_%s_pT0150_E_scheme",kTracksName.Data()));
TString kTpcJetsR04EschemeName(Form("Jet_AKTChargedR040_%s_pT0150_E_scheme",kTracksName.Data()));
TString kTpcJetsR05EschemeName(Form("Jet_AKTChargedR050_%s_pT0150_E_scheme",kTracksName.Data()));
TString kTpcJetsR07EschemeName(Form("Jet_AKTChargedR070_%s_pT0150_E_scheme",kTracksName.Data()));
TString kTpcJetsR07EschemeName2(Form("Jet_AKTChargedR070_%s_pT0150_E_scheme",kTracksName2.Data()));
TString kTpcJetsR02BIpTschemeName(Form("Jet_AKTChargedR020_%s_pT0150_BIpt_scheme",kTracksName.Data()));
TString kTpcJetsR03BIpTschemeName(Form("Jet_AKTChargedR030_%s_pT0150_BIpt_scheme",kTracksName.Data()));
TString kTpcJetsR04BIpTschemeName(Form("Jet_AKTChargedR040_%s_pT0150_BIpt_scheme",kTracksName.Data()));

TString kChargedJetsR02Name("Jet_AKTChargedR020_tracks_pT0150_pt_scheme");
TString kChargedJetsR03Name("Jet_AKTChargedR030_tracks_pT0150_pt_scheme");
TString kChargedJetsR04Name("Jet_AKTChargedR040_tracks_pT0150_pt_scheme");
TString kChargedJetsR06Name("Jet_AKTChargedR060_tracks_pT0150_pt_scheme");
TString kFullJetsR02Name("Jet_AKTFullR020_tracks_pT0150_caloClusters_ET0300_pt_scheme");
TString kFullJetsR04Name("Jet_AKTFullR040_tracks_pT0150_caloClusters_ET0300_pt_scheme");
TString kFullJetsR06Name("Jet_AKTFullR060_tracks_pT0150_caloClusters_ET0300_pt_scheme");

// Jet settings, change with care
Int_t kLeadHadType = 0; // 0 = charged, 1 = neutral, 2 = both
Int_t kHistoType = 1; // 1 = THnSparse, 0 = TH1/TH2/TH3
Double_t kGhostArea = 0.01;
Double_t kKtJetRadius = 0.4;
Double_t kJetAreaCut = 0.557;
Double_t kJetPtCut = 1;
Double_t kJetBiasTrack = 5;
Double_t kJetBiasClus = 1000;
Double_t kClusPtCut = 0.30;
Double_t kTrackPtCut = 0.15;
Double_t kPartLevPtCut = 0;
Double_t kHadCorr = 2.; //1.7;//
Double_t kTrackEff = 1.0; //0.96;//


/*************************************/
// ### Settings for Ruediger

Bool_t kRHDoRandomize = kFALSE;
Bool_t kRHImportEventPool = kTRUE;
Bool_t kRHExportEventPool = kFALSE;
//Bool_t kRHImportEventPool = kFALSE;
//Bool_t kRHExportEventPool = kTRUE;


// ### Variables
TString kTracksRH = "";
if(kRHDoRandomize) kTracksRH = "tracks_randomized"; else kTracksRH = "tracks";
TString kJetsAKTR030RH_015 = Form("Jet_AKTChargedR030_%s_pT%04i_pt_scheme", kTracksRH.Data(), (Int_t)(1000*0.15));
TString kJetsAKTR030RH_1 = Form("Jet_AKTChargedR030_%s_pT%04i_pt_scheme", kTracksRH.Data(), (Int_t)(1000*1.0));
TString kJetsAKTR030RH_2 = Form("Jet_AKTChargedR030_%s_pT%04i_pt_scheme", kTracksRH.Data(), (Int_t)(1000*2.0));
TString kJetsAKTR030RH_3 = Form("Jet_AKTChargedR030_%s_pT%04i_pt_scheme", kTracksRH.Data(), (Int_t)(1000*3.0));

TString kRHBinning = "";
kRHBinning = "p_t_assoc: 0.15, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0, 60.0\n

delta_phi: -1.5708, -1.45664, -1.34248, -1.22832, -1.11416, -1, -0.9, -0.8, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.275, -0.25, -0.225, -0.2, -0.175, -0.15, -0.125, -0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1, 1.11416, 1.22832, 1.34248, 1.45664, 1.5708, 1.68496, 1.79911, 1.91327, 2.02743, 2.14159, 2.19159, 2.24159, 2.29159, 2.34159, 2.39159, 2.44159, 2.49159, 2.54159, 2.59159, 2.64159, 2.69159, 2.74159, 2.79159, 2.84159, 2.89159, 2.94159, 2.99159, 3.04159, 3.09159, 3.14159, 3.19159, 3.24159, 3.29159, 3.34159, 3.39159, 3.44159, 3.49159, 3.54159, 3.59159, 3.64159, 3.69159, 3.74159, 3.79159, 3.84159, 3.89159, 3.94159, 3.99159, 4.04159, 4.09159, 4.14159, 4.25575, 4.36991, 4.48407, 4.59823, 4.71239\n
delta_eta: -2.4, -2.0, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.275, -0.25, -0.225, -0.2, -0.175, -0.15, -0.125, -0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.4
p_t_leading: 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 120.0 \np_t_leading_course: 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 120.0 \n";

/*************************************/

/*
// Some useful constants
Double_t kPi = TMath::Pi();
UInt_t kTypeFullAKTR04 = 1<<1 | 1<<2 | 1<<7;
*/

// Vojtech
TF1* sfunc = new TF1("sfunc","[0]*x*x+[1]*x+[2]",-1,100); // for V0s_AliAnalysisTaskRhoTpcExLJ wagon
sfunc->SetParameter(2, 1.81208);
sfunc->SetParameter(1, -0.0105506);
sfunc->SetParameter(0, 0.000145519);
const char* kPeriod = "lhc11a";
const char* kTrackCuts = "Hybrid_LHC11a";
const char* kDatatype = "AOD";
TString kOrigClusName = "caloClusters";
TString kEMCALCellsName="emcalCells";
//TString kInTracksName = "AODFilterTracks";
AliTrackContainer::SetDefTrackCutsPeriod("lhc11a");
}
