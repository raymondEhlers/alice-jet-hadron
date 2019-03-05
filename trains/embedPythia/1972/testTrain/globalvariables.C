{
/*************************************/
/* Some definitions                  */
/*************************************/
// Dec 11, 2015 - this parameter undefined from AddTaskEmcalClusterMaker
// DO NOT UNCOMMENT: this parameter should be defined in the global variables of each period!!!
//const UInt_t kNonLinFunct  = AliEMCALRecoUtils::kBeamTestCorrected;

// Necessary in new jet framework
// DO NOT UNCOMMENT: this parameter should be defined in the global variables of each period!!!
//AliTrackContainer::SetDefTrackCutsPeriod("lhc11h"); // for correct hybrid track selection/track tagging

/*************************************/

/*************************************/
/* Trigger & Centralities            */
/*************************************/

Int_t nCentBins = 4;
Double_t minCent = 0;
Double_t maxCent = 100;

UInt_t kComPhysSel = AliVEvent::kAny;

//UInt_t kComPhysSel = AliVEvent::kINT7 + AliVEvent::kEMCEGA + AliVEvent::kEMCEJE + AliVEvent::kEMC7;

UInt_t kPhysSel = 1<<31; //AliEmcalPhysicsSelection::kEmcalOk
//UInt_t kPhysSel = AliVEvent::kAnyINT + AliVEvent::kSemiCentral + AliVEvent::kCentral;
//UInt_t kPhysSel = AliVEvent::kAny;
//UInt_t kPhysSel = AliVEvent::kMB;
//UInt_t kPhysSel = AliVEvent::kEMCEGA;
//UInt_t kPhysSelCentral = AliVEvent::kCentral;
//UInt_t kComPhysSel = AliVEvent::kMB + AliVEvent::kSemiCentral + AliVEvent::kCentral + AliVEvent::kEMCEGA + AliVEvent::kEMCEJE;
//UInt_t kComPhysSel = AliVEvent::kAny;
//UInt_t kComPhysSel = AliVEvent::kEMCEGA;
//UInt_t kComPhysSel = AliVEvent::kEMCEJE;
//UInt_t kComPhysSel = AliVEvent::kMB + AliVEvent::kSemiCentral + AliVEvent::kCentral;
//UInt_t kComPhysSel = AliVEvent::kMB + AliVEvent::kCentral;
//UInt_t kComPhysSel = AliVEvent::kEMCEJE;
//UInt_t kComPhysSel = AliVEvent::kMB;

// Some trigger definitions for Joel
UInt_t kPhysSelJetHadEMCEJE = AliVEvent::kMB + AliVEvent::kSemiCentral + AliVEvent::kCentral + AliVEvent::kEMCEJE;
UInt_t kPhysSelJetHadEMCEJGA = AliVEvent::kMB + AliVEvent::kSemiCentral + AliVEvent::kCentral + AliVEvent::kEMCEJE + AliVEvent::kEMCEGA;
UInt_t kPhysSelJetHadEMCEGA = AliVEvent::kMB + AliVEvent::kSemiCentral + AliVEvent::kCentral + AliVEvent::kEMCEGA;
UInt_t kPhysSelJetHadMB = AliVEvent::kMB + AliVEvent::kSemiCentral + AliVEvent::kCentral;


/*************************************/
// Some definitions for Ruediger

// ### Settings
Double_t kRHMinLeadingHadronPt = 10.0; // only applies to leading hadron biased data
Double_t kRHMaxLeadingHadronPt = 100.;
Double_t kRHTrackPtCut = 0.15; // constituent cut for the jet finder
Bool_t kRHUseFakefactorRejection = kFALSE;
Bool_t kRHDoRandomize = kFALSE;
Bool_t kRHExportEventPool = kTRUE;
Bool_t kRHImportEventPool = kFALSE;

// Dijet trigger
Bool_t kRHUseDijetTrigger = kFALSE;
Int_t kRHDijetTriggerMode = 1; // 1: leading, 2: subleading, 3: all
Double_t kRHDijetTriggerLeadingPt = 20.;
Double_t kRHDijetTriggerSubleadingPt = 20.;

const char* kRHRhoName = "RhoR020KT";//RhoR020KT
TString kRHCustomBinning = "";
kRHCustomBinning = "p_t_assoc: 0.15, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0\n

delta_phi: -1.5708, -1.45664, -1.34248, -1.22832, -1.11416, -1, -0.9, -0.8, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.275, -0.25, -0.225, -0.2, -0.175, -0.15, -0.125, -0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1, 1.11416, 1.22832, 1.34248, 1.45664, 1.5708, 1.68496, 1.79911, 1.91327, 2.02743, 2.14159, 2.19159, 2.24159, 2.29159, 2.34159, 2.39159, 2.44159, 2.49159, 2.54159, 2.59159, 2.64159, 2.69159, 2.74159, 2.79159, 2.84159, 2.89159, 2.94159, 2.99159, 3.04159, 3.09159, 3.14159, 3.19159, 3.24159, 3.29159, 3.34159, 3.39159, 3.44159, 3.49159, 3.54159, 3.59159, 3.64159, 3.69159, 3.74159, 3.79159, 3.84159, 3.89159, 3.94159, 3.99159, 4.04159, 4.09159, 4.14159, 4.25575, 4.36991, 4.48407, 4.59823, 4.71239\n
delta_eta: -2.4, -2.0, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.275, -0.25, -0.225, -0.2, -0.175, -0.15, -0.125, -0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.4
p_t_leading: 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0 \np_t_leading_course: 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0 \n";


// ### Variables
UInt_t kPhysSelRH = AliVEvent::kMB + AliVEvent::kSemiCentral + AliVEvent::kCentral;

TString kTracksRH = "";
if(kRHDoRandomize) kTracksRH = "tracks_randomized"; else kTracksRH = "tracks";
TString kJetsAKTR040RH = Form("Jet_AKTChargedR040_%s_pT%04d_pt_scheme", kTracksRH.Data(), (Int_t)(kRHTrackPtCut*1000));
TString kJetsAKTR030RH = Form("Jet_AKTChargedR030_%s_pT%04d_pt_scheme", kTracksRH.Data(), (Int_t)(kRHTrackPtCut*1000));
TString kJetsAKTR020RH = Form("Jet_AKTChargedR020_%s_pT%04d_pt_scheme", kTracksRH.Data(), (Int_t)(kRHTrackPtCut*1000));
TString kJetsKTR020RH = Form("Jet_KTChargedR020_%s_pT%04d_pt_scheme", kTracksRH.Data(), (Int_t)(kRHTrackPtCut*1000));
TString kJetsKTR020RHEmbedded = Form("Jet_KTChargedR020_%s_pT%04d_pt_scheme", "tracks_generated_embedded", (Int_t)(kRHTrackPtCut*1000));
TString kJetsAKTR030RHEmbedded = Form("JetEmb_AKTChargedR030_%s_pT%04d_pt_scheme", "tracks_generated_embedded", (Int_t)(kRHTrackPtCut*1000));
TString kJetsAKTR030RHEmbeddedPbPb = Form("Jet_AKTChargedR030_%s_pT%04d_pt_scheme", "tracks", (Int_t)(kRHTrackPtCut*1000));
TString kJetsAKTR030RHPYTHIA = Form("JetPY_AKTChargedR030_%s_pT%04d_pt_scheme", "tracks_generated_PY", (Int_t)(kRHTrackPtCut*1000));
TString kTracksRHJetRemoval = Form("tracks_jetremoval");
TString kJetsAKTR030RHJetRemoval = Form("JetJR_AKTChargedR030_%s_pT%04d_pt_scheme", kTracksRHJetRemoval.Data(), (Int_t)(kRHTrackPtCut*1000));


/*************************************/

/*************************************/

/*************************************/
/* Analysis Cuts & Settings          */
/*************************************/

Bool_t kDoAODTrackProp = kTRUE;
Bool_t kDoTender = kTRUE;
Double_t kClusPtCut = 0.30;
Double_t kTrackPtCut = 0.15;
Double_t kMinPtHadCorr = 0.15;

Bool_t kMakeEmcalTriggers = kFALSE;
const char* kEmcalTriggers = "";
if(kMakeEmcalTriggers) kEmcalTriggers = "EmcalTriggers";

Double_t kTrackEff = 1.0;

Double_t kPropDist = 440;
Double_t kHadCorr = 2.0;
Bool_t kDoEmbedding = kFALSE;
Bool_t kDoReclusterize = kTRUE;
if (kDoEmbedding) kDoReclusterize = kTRUE;
UInt_t kClusterizerType = AliEMCALRecParam::kClusterizerv2;

Double_t kEMCtimeMin = -50e-6;
Double_t kEMCtimeMax = 100e-6;
Double_t kEMCtimeCut = 75e-6;

Double_t kGhostArea = 0.005;
Double_t kKtJetRadius = 0.2;
Double_t kJetAreaCut = 0.;

Double_t kJetPtCut = 1;
Double_t kJetBiasTrack = 5;
Double_t kJetBiasClus = 1000;

Double_t kRhoMinEta = -0.5;
Double_t kRhoMaxEta = 0.5;
Double_t kRhoMinPhi = 30*TMath::DegToRad()+0.2;
Double_t kRhoMaxPhi = 230*TMath::DegToRad()-0.2;

Double_t kEmcalMinEta = -0.7;
Double_t kEmcalMaxEta = 0.7;
Double_t kEmcalMinPhi = 80*TMath::DegToRad();
Double_t kEmcalMaxPhi = 180*TMath::DegToRad();

Int_t kLeadHadType = 1; // 0 = charged, 1 = neutral, 2 = both
Int_t kHistoType = 1; // 1 = THnSparse, 0 = TH1/TH2/TH3

// Apply Level1 Phase Shift correction in LHC15o data
Bool_t kApplyL1PhaseShiftCorrection = kFALSE;

/*************************************/

/*************************************/
/* PYTHIA Embedding                  */
/*************************************/


Double_t kJetLeadingTrackBias = 0;
Int_t kNcent = 3;

Int_t kNjetResp = 1;
if (kJetLeadingTrackBias > 1)
  kNjetResp = 2;

if (kJetLeadingTrackBias > 5)
  kNjetResp = 3;
kNjetResp *= kNcent;

Bool_t kMakeTrigger = kFALSE;

if (1) {
  UInt_t kPythiaR020Charged = 0;
  UInt_t kPythiaR030Charged = 0;
  UInt_t kPythiaR040Charged = 0;

  UInt_t kPythiaR020Full = 0;
  UInt_t kPythiaR030Full = 0;
}
// Code not compiling - please leave this commented out!
//else {
//  UInt_t kPythiaR020Charged = kAKT|kR020Jet|kChargedJet;
//  UInt_t kPythiaR030Charged = kAKT|kR030Jet|kChargedJet;
//  UInt_t kPythiaR040Charged = kAKT|kR040Jet|kChargedJet;
//
//  UInt_t kPythiaR020Full = kAKT|kR020Jet|kFullJet;
//  UInt_t kPythiaR030Full = kAKT|kR030Jet|kFullJet;
//}

TString kPYTHIAPath = "alien:///alice/sim/2012/LHC12a15e_fix/%d/%d/AOD149/%04d/AliAOD.root";
Int_t nAODFiles = 140;
Int_t kNpTHardBins = 11;

Double_t kPtHardBinsScaling[11] = {0, 0, 3.27317e-05, 2.57606e-06, 2.5248e-07, 2.92483e-08, 4.15631e-09, 6.6079e-10, 1.49042e-10, 3.5571e-11, 1.29474e-11};

//Double_t kPtHardBinsScaling[11] = {0.000000E+00, 5.206101E-05, 5.859497E-06, 4.444755E-07, 4.344664E-08, 5.154750E-09, 6.956634E-10, 1.149828E-10, 2.520137E-11, 6.222240E-12, 2.255832E-12};


//kPtHardBinsScaling[6] = 1;
//kPtHardBinsScaling[1] = 0;
//kPtHardBinsScaling[2] = 0;

//Separate train for each pt hard bin
for(Int_t i =1; i<11; i++) kPtHardBinsScaling[i] = 0;
Int_t kPtHardBin = 4; // set here the bin
kPtHardBinsScaling[kPtHardBin] = 1;


// Uncomment the line below to have a flat distribution of pt hard bins
//for(Int_t i =1; i<11; i++) kPtHardBinsScaling[i] = 1;

Double_t kMinPythiaJetPt = 0;//20; //changed by Marta

/*************************************/
/* Containers' names                 */
/*************************************/

TString kTracksName = "tracks";
TString kClusName = "EmcCaloClusters";
TString kCorrClusName = "CaloClustersCorr";
TString kEmcalTracksName = "EmcalTracks_";
kEmcalTracksName += kTracksName;
TString kEmcalClusName = "EmcalClusters_";
kEmcalClusName += kClusName;


// TEST name schemes:
// added July13th for memory tests - KEEP
TString kTracksNameTEST = "AODFilterTracks"; // "PicoTracks";
TString kEmcalTracksNameTEST = "EmcalTracks_";
kEmcalTracksNameTEST += kTracksNameTEST;
TString kTpcRhoNameTEST = "TpcRho";
TString kTpcRhoNameExLJTEST = "TpcRho_ExLJ";
TString kTpcRhoNameExLJTEST1 = "TpcRho_ExLJ1"; // for 1+ GeV constituent jets
TString kTpcRhoNameExLJTEST2 = "TpcRho_ExLJ2"; // for 2+ GeV constituent jets
TString kTpcRhoNameExLJTEST3 = "TpcRho_ExLJ3"; // for 3+ GeV constituent jets
TString kTpcKtJetsNameTEST(Form("Jet_KTChargedR020_%s_pT0150_pt_scheme",kTracksNameTEST.Data()));
TString kTpcJetsR02NameTEST(Form("Jet_AKTChargedR020_%s_pT0150_pt_scheme",kTracksNameTEST.Data()));
TString kTpcRhoExLJScaledNameTEST = "TpcRho_ExLJ_Scaled";

// **** noticed on May4, 2016 - for below strings, there is no more "ET" is now "E" for scheme of jet name
TString kEmcalJets3GeVR02NameTEST(Form("Jet_AKTFullR020_%s_pT3000_%s_E3000_pt_scheme",kTracksNameTEST.Data(),kCorrClusName.Data()));
TString kEmcalJets3GeVR02NameTEST2(Form("Jet_AKTFullR020_%s_pT3000_%s_E3000_pt_scheme",kTracksNameTEST.Data(),kClusName.Data()));
TString kEmcalJets5GeVR02NameTEST2(Form("Jet_AKTFullR020_%s_pT5000_%s_E5000_pt_scheme",kTracksNameTEST.Data(),kClusName.Data()));
TString kEmcalJets3GeVR01NameTEST(Form("Jet_AKTFullR010_%s_pT3000_%s_E3000_pt_scheme",kTracksNameTEST.Data(),kCorrClusName.Data()));

// NEW name scheme for framework changes - Dec 12, 2015
TString kTracksNameNEW = "tracks";
TString kTpcKtJetsNameNEW(Form("Jet_KTChargedR020_%s_pT0150_pt_scheme",kTracksNameNEW.Data()));
TString kTpcKtJetsNameNEW1(Form("Jet_KTChargedR020_%s_pT1000_pt_scheme",kTracksNameNEW.Data()));
TString kTpcKtJetsNameNEW2(Form("Jet_KTChargedR020_%s_pT2000_pt_scheme",kTracksNameNEW.Data()));
TString kTpcKtJetsNameNEW3(Form("Jet_KTChargedR020_%s_pT3000_pt_scheme",kTracksNameNEW.Data()));
TString kTpcJetsR02NameNEW(Form("Jet_AKTChargedR020_%s_pT0150_pt_scheme",kTracksNameNEW.Data()));



TString kClusRandName = "CaloClustersCorrRandomized";
TString kClusEmbName = "CaloClustersCorrEmbedded";

TString kTrackRandName = "PicoTracksRandomized";
TString kTrackEmbName = "PicoTracksEmbedded";
TString kTrackEmcalEmbName = "PicoTracksEmcalEmbedded";
TString kTrackEmbSpectrumName = "PicoTracksEmbeddedSpectrum";

TString kTpcRhoName = "TpcRho";
TString kTpcRhoNameExLJ = "TpcRho_ExLJ";
TString kTpcRhoSmallName = "TpcRho_Small";
TString kEmcalRhoMeth2Name = "TpcRho_Small_Scaled";
TString kTpcRhoExLJScaledName = "TpcRho_ExLJ_Scaled";

TString kTpcKtJetsName(Form("Jet_KTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcKtJetsR04Name(Form("Jet_KTChargedR040_%s_pT0150_pt_scheme",kTracksName.Data()));

TString kTpcJetsR01Name(Form("Jet_AKTChargedR010_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR02Name(Form("Jet_AKTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR03Name(Form("Jet_AKTChargedR030_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR04Name(Form("Jet_AKTChargedR040_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcEmbJetsR02Name(Form("Jet_AKTChargedR020_%s_pT0150_pt_scheme",kTrackEmbName.Data()));
TString kTpcEmbJetsR03Name(Form("Jet_AKTChargedR030_%s_pT0150_pt_scheme",kTrackEmbName.Data()));
TString kTpcEmbJetsR04Name(Form("Jet_AKTChargedR040_%s_pT0150_pt_scheme",kTrackEmbName.Data()));

TString kEmcalKtJetsName(Form("Jet_KTFullR020_%s_pT0150_%s_ET0300_pt_scheme",kTracksName.Data(),kCorrClusName.Data()));

TString kEmcalJets01Name(Form("Jet_AKTFullR010_%s_pT0150_%s_ET0%d_pt_scheme",kTracksName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalJets015Name(Form("Jet_AKTFullR015_%s_pT0150_%s_ET0%d_pt_scheme",kTracksName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalJets02Name(Form("Jet_AKTFullR020_%s_pT0150_%s_ET0%d_pt_scheme",kTracksName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalJets025Name(Form("Jet_AKTFullR025_%s_pT0150_%s_ET0%d_pt_scheme",kTracksName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalJets03Name(Form("Jet_AKTFullR030_%s_pT0150_%s_ET0%d_pt_scheme",kTracksName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalJets04Name(Form("Jet_AKTFullR040_%s_pT0150_%s_ET0%d_pt_scheme",kTracksName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));

TString kNeutralRhoName = "NeutralRho";
TString kNeutralEMCalRhoName = "NeutralRhoEMCal";
TString kNeutralDCalRhoName = "NeutralRhoDCal";
TString kNeutralKtJets02Name(Form("Jet_KTFullR020_%s_ET0%d_pt_scheme","CaloClusters",TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kNeutralJets02Name(Form("Jet_AKTFullR020_%s_ET0%d_pt_scheme","CaloClusters",TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kNeutralJets04Name(Form("Jet_AKTFullR040_%s_ET0%d_pt_scheme","CaloClusters",TMath::FloorNint(kClusPtCut*1000+0.5)));

TString kEmalJet02UncorrName = "Jet_AKTNeutralR020_EmcCaloClusters_ET0300_pt_scheme";

TString kEmcalEmbJets015Name(Form("Jet_AKTFullR015_%s_pT0150_%s_ET0%d_pt_scheme",kTrackEmcalEmbName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalEmbJets02Name(Form("Jet_AKTFullR020_%s_pT0150_%s_ET0%d_pt_scheme",kTrackEmcalEmbName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalEmbJets025Name(Form("Jet_AKTFullR025_%s_pT0150_%s_ET0%d_pt_scheme",kTrackEmcalEmbName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalEmbJets03Name(Form("Jet_AKTFullR030_%s_pT0150_%s_ET0%d_pt_scheme",kTrackEmcalEmbName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));
TString kEmcalEmbJets04Name(Form("Jet_AKTFullR040_%s_pT0150_%s_ET0%d_pt_scheme",kTrackEmcalEmbName.Data(),kCorrClusName.Data(),TMath::FloorNint(kClusPtCut*1000+0.5)));

// 1GeV constituent cut
TString kTpcKtJets1GeVName(Form("Jet_KTChargedR020_%s_pT1000_pt_scheme",kTracksName.Data()));
TString kTpcRho1GeVNameExLJ = "TpcRho_1GeV_ExLJ";
TString kTpcRho1GeVSmallName = "TpcRho_1GeV_Small";
TString kEmcalRho1GeVMeth2Name = "TpcRho_1GeV_Small_Scaled";
TString kTpcJets1GeVR02Name(Form("Jet_AKTChargedR020_%s_pT1000_pt_scheme",kTracksName.Data()));
TString kTpcJets1GeVR03Name(Form("Jet_AKTChargedR030_%s_pT1000_pt_scheme",kTracksName.Data()));
TString kEmcalJets1GeVR02Name(Form("Jet_AKTFullR020_%s_pT1000_%s_ET1000_pt_scheme",kTracksName.Data(),kCorrClusName.Data()));
TString kEmcalJets1GeVR03Name(Form("Jet_AKTFullR030_%s_pT1000_%s_ET1000_pt_scheme",kTracksName.Data(),kCorrClusName.Data()));

// 3GeV constituent cut
TString kTpcJets3GeVR02Name(Form("Jet_AKTChargedR020_%s_pT3000_pt_scheme",kTracksName.Data()));
TString kTpcJets3GeVR03Name(Form("Jet_AKTChargedR030_%s_pT3000_pt_scheme",kTracksName.Data()));
TString kEmcalJets3GeVR02Name(Form("Jet_AKTFullR020_%s_pT3000_%s_ET3000_pt_scheme",kTracksName.Data(),kCorrClusName.Data()));
TString kEmcalJets3GeVR03Name(Form("Jet_AKTFullR030_%s_pT3000_%s_ET3000_pt_scheme",kTracksName.Data(),kCorrClusName.Data()));

// Toy model Single Particle embedding
TString kTpcEmbSpectrumJets02Name(Form("Jet_AKTChargedR020_%s_pT0150_pt_scheme",kTrackEmbSpectrumName.Data()));
TString kTpcEmbSpectrumJets03Name(Form("Jet_AKTChargedR030_%s_pT0150_pt_scheme",kTrackEmbSpectrumName.Data()));
TString kTpcEmbSpectrumJets04Name(Form("Jet_AKTChargedR040_%s_pT0150_pt_scheme",kTrackEmbSpectrumName.Data()));

//Jets with E recombination scheme (added by Marta)
TString kTpcKtJetsESchemeName(Form("Jet_KTChargedR020_%s_pT0150_E_scheme",kTracksNameNEW.Data()));
TString kTpcJetsR02ESchemeName(Form("Jet_AKTChargedR020_%s_pT0150_E_scheme",kTracksNameNEW.Data()));
TString kTpcJetsR04ESchemeName(Form("Jet_AKTChargedR040_%s_pT0150_E_scheme",kTracksNameNEW.Data()));
TString kNeutralKtJetsESchemeName(Form("Jet_KTNeutralR020_%s_ET0300_E_scheme",kCorrClusName.Data()));

//Jets with E recombination scheme (added by Davide&Leticia)
TString kTpcKtJetsESchemeName1GeV(Form("Jet_KTChargedR020_%s_pT1000_E_scheme",kTracksName.Data()));
TString kTpcKtJetsESchemeNameEmb(Form("Jet_KTChargedR020_%s_pT0150_E_scheme","PicoTracksEmb"));
TString kTpcAKtJetsESchemeNameEmb(Form("Jet_AKTChargedR040_%s_pT0150_E_scheme","PicoTracksEmb"));
TString kTpcAKtJetsMCESchemeNameEmb(Form("JetMCOnly_AKTChargedR040_%s_pT0150_E_scheme","PicoTracksEmb"));
Double_t kPtQG=0.15;

Double_t kHolePos=0;
Double_t kHoleWidth=0;
Int_t kSemigoodRun=0;

//IROC13
//Double_t kHolePos=4.71;
//Double_t kHoleWidth=0.35;
//Int_t kSemigoodRun=1;

//OROC08
//Double_t kHolePos=2.97;
//Double_t kHoleWidth=0.7;
//Int_t kSemigoodRun=1;


/*************************************/

/*************************************/
/* Trigger Tasks                     */
/*************************************/

TString kTriggerClusName = "L1TriggerClusters";
TString kTriggerClusFastORName = "L1TriggerClustersFastOR";
TF1 *kTriggerThreshold = new TF1("eth","[0] + [1]*x + [2]*x*x",0,100);
//kTriggerThreshold->SetParameter(0,137.8);
//kTriggerThreshold->SetParameter(1,-1.28);
//kTriggerThreshold->SetParameter(0,79.9);
//kTriggerThreshold->SetParameter(1,-0.701);
kTriggerThreshold->SetParameter(0,5.);
kTriggerThreshold->SetParameter(1,0.0031);
kTriggerThreshold->SetParameter(2,0.);


Int_t kManual = 0, kEmcalJet = 1;

/*************************************/

// Containers' names
TString kMCTracksName = "MCSelectedParticles";
const char* kUsedMCParticles = "MCParticlesSelected";

TString kMCChargedJetsR02Name(Form("Jet_AKTChargedR020_%s_pT0000_pt_scheme",kMCTracksName.Data()));
TString kMCChargedJetsR03Name(Form("Jet_AKTChargedR030_%s_pT0000_pt_scheme",kMCTracksName.Data()));
TString kMCChargedJetsR04Name(Form("Jet_AKTChargedR040_%s_pT0000_pt_scheme",kMCTracksName.Data()));
TString kMCFullJetsR02Name(Form("Jet_AKTFullR020_%s_pT3000_pt_scheme",kMCTracksName.Data()));
TString kMCFullJetsR03Name(Form("Jet_AKTFullR030_%s_pT0000_pt_scheme",kMCTracksName.Data()));

//Exclude embedding
TString kOrigClusExcludeEmbName = "CaloClustersExcludeEmb";
TString kClusExcludeEmbName = "EmcCaloClustersExcludeEmb";
TString kCorrClusExcludeEmbName = "CaloClustersCorrExcludeEmb";
TString kEmcalClusExcludeEmbName = "EmcalClusters_";
kEmcalClusExcludeEmbName += kClusExcludeEmbName;
TString kEmcalTracksExcludeEmbName = "EmcalTracks_";
kEmcalTracksExcludeEmbName += "ExcludeEmb";
kEmcalTracksExcludeEmbName += kTracksName;

TString kTpcJetsR02ExcludeEmbName(Form("JetExcludeEmb_AKTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR03ExcludeEmbName(Form("JetExcludeEmb_AKTChargedR030_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR04ExcludeEmbName(Form("JetExcludeEmb_AKTChargedR040_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kEmcalJetsR02ExcludeEmbName(Form("JetExcludeEmb_AKTFullR020_%s_pT0150_%s_ET0300_pt_scheme",kTracksName.Data(),kCorrClusExcludeEmbName.Data()));
TString kEmcalJetsR03ExcludeEmbName(Form("JetExcludeEmb_AKTFullR030_%s_pT0150_%s_ET0300_pt_scheme",kTracksName.Data(),kCorrClusExcludeEmbName.Data()));

TString kTpcKtJetsExcludeEmbName(Form("JetExcludeEmb_KTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));

TString kTpcRhoExLJExcludeEmbName = "TpcRho_ExLJ_ExcludeEmb";
TString kTpcRhoSmallExcludeEmbName = "TpcRho_Small_ExcludeEmb";
TString kEmcalRhoMeth2ExcludeEmbName = "TpcRho_Small_ExcludeEmb_Scaled";

//Embedding only
TString kOrigClusEmbOnlyName = "CaloClustersEmbOnly";
TString kClusEmbOnlyName = "EmcCaloClustersEmbOnly";
TString kCorrClusEmbOnlyName = "CaloClustersCorrEmbOnly";
TString kEmcalClusEmbOnlyName = "EmcalClusters_";
kEmcalClusEmbOnlyName += kClusEmbOnlyName;
TString kEmcalTracksEmbOnlyName = "EmcalTracks_";
kEmcalTracksEmbOnlyName += "EmbOnly";
kEmcalTracksEmbOnlyName += kTracksName;

TString kTpcJetsR02EmbOnlyName(Form("JetEmbOnly_AKTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR03EmbOnlyName(Form("JetEmbOnly_AKTChargedR030_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kTpcJetsR04EmbOnlyName(Form("JetEmbOnly_AKTChargedR040_%s_pT0150_pt_scheme",kTracksName.Data()));
TString kEmcalJetsR02EmbOnlyName(Form("JetEmbOnly_AKTFullR020_%s_pT0150_%s_ET0300_pt_scheme",kTracksName.Data(),kCorrClusEmbOnlyName.Data()));
TString kEmcalJetsR03EmbOnlyName(Form("JetEmbOnly_AKTFullR030_%s_pT0150_%s_ET0300_pt_scheme",kTracksName.Data(),kCorrClusEmbOnlyName.Data()));

//PbPb only
TString kTpcJetsR02PbPbOnlyName(Form("JetPbPbOnly_AKTChargedR020_%s_pT0150_pt_scheme",kTracksName.Data()));

// Other settings
UInt_t kMatching = 1; //1=geometrical, 2=MClabel, 3=same collections

Double_t kMaxDistance02a = 1;
Double_t kMaxDistance03a = 1;
Double_t kMaxDistance04a = 1;
Double_t kMaxDistance02b = 1;
Double_t kMaxDistance03b = 1;
Double_t kMaxDistance04b = 1;
Double_t kMaxGeoDistance02 = 1.2;
Double_t kMaxGeoDistance03 = 1.2;
Double_t kMaxGeoDistance04 = 1.2;

Double_t kPartLevPtCut = 3;

/*************************************/

/*************************************/
/* Redmer                            */
/*************************************/

AliAnalysisTaskSE* kRhoVnMod[10];

/*************************************/

/*************************************/
/* Darius                            */
/*************************************/

/*************************************/
/* Marta                            */
/*************************************/
TString kRhoMassName = "RhoMass";
Float_t kChJetPhiMin = -1;
Float_t kChJetPhiMax = -1;
Float_t kChJetPhiOffset = 0.;

TF1* srhomfunc = new TF1("srhomfunc","[0]*x*x+[1]*x+[2]",-1,100);
srhomfunc->SetParameter(2, 1.68354);
srhomfunc->SetParameter(1, -2.86991e-03);
srhomfunc->SetParameter(0, -1.49981e-05);

/*************************************/
/* Scale Factors                     */
/*************************************/

TF1* sfunc = new TF1("sfunc","[0]*x*x+[1]*x+[2]",-1,100);
//TF1* sfunc = NULL;

TF1* sfunc1GeV = new TF1("sfunc","[0]*x*x+[1]*x+[2]",-1,100);

if (kHadCorr > 1.9) { // had corr = 2
  if (kClusterizerType==AliEMCALRecParam::kClusterizerv2) { // v2 clusterizer
    // used for train n. 374
    sfunc->SetParameter(2, 1.76458);
    sfunc->SetParameter(1, -0.0111656);
    sfunc->SetParameter(0, 0.000107296);
    // fit from full train n. 374
    //sfunc->SetParameter(2, 1.78067);
    //sfunc->SetParameter(1, -0.0116032);
    //sfunc->SetParameter(0, 0.000114951);

    // fit from test train n. 457
    sfunc1GeV->SetParameter(2, 1.63498);
    sfunc1GeV->SetParameter(1, -0.0208189);
    sfunc1GeV->SetParameter(0, 0.000303435);
  } // end v2 clusterizer
  else { // 3x3 clusterizer
    // used for train n. 408 (fit test train n.408)
    //sfunc->SetParameter(2, 1.85043);
    //sfunc->SetParameter(1, -0.0116461);
    //sfunc->SetParameter(0, 9.65308e-05);
    // fit from full train n.408
    sfunc->SetParameter(2, 1.8528);
    sfunc->SetParameter(1, -0.0121015);
    sfunc->SetParameter(0, 0.000109308);
  } // end 3x3 clusterizer
} // end had corr = 2.0
else if (kHadCorr > 1.6) { // had corr = 1.7
  // fit from test run of train n. 387
  sfunc->SetParameter(2, 1.8171);
  sfunc->SetParameter(1, -0.0150141);
  sfunc->SetParameter(0, 0.000193964);
} // end had corr = 1.7
else if (kHadCorr > 1.2) { // had corr = 1.3
  // fit from test run of train n. 389
  sfunc->SetParameter(2, 1.90496);
  sfunc->SetParameter(1, -0.016978);
  sfunc->SetParameter(0, 0.000215038);
} // end had corr = 1.3
else {   // had corr = 0
  if (kClusterizerType==AliEMCALRecParam::kClusterizerv2) { // v2 clusterizer
    // 300 MeV, had corr = 2
    // fit from full train n. 401
    sfunc->SetParameter(2, 2.0418);
    sfunc->SetParameter(1, -0.0157904);
    sfunc->SetParameter(0, 0.000148058);
  } // end v2 clusterizer
  else { // 3x3 clusterizer
    // fit from full train n. 407
    sfunc->SetParameter(2, 2.13631);
    sfunc->SetParameter(1, -0.0165245);
    sfunc->SetParameter(0, 0.000142767);
  } // end 3x3 clusterizer
} // end had corr = 0

if (0) { // GA trigger
  if (kHadCorr > 1.9) { // had corr = 2
    // old
    //sfunc->SetParameter(2,1.92213);
    //sfunc->SetParameter(1,-0.0175260);
    //sfunc->SetParameter(0,0.000152385);

    sfunc->SetParameter(2, 1.81208);
    sfunc->SetParameter(1, -0.0105506);
    sfunc->SetParameter(0, 0.000145519);
  } // end had corr = 2.0
} //end ga trigger

/*************************************/


//std::cout << "_____________________________________________________________________" << std::endl << "Printing all global variables" << //std::endl << "_____________________________________________________________________" << std::endl;
//gSystem->Exec("printenv");
//std::cout << "_____________________________________________________________________" << std::endl;

/*************************************/
const char* kPeriod = "lhc11h";
const char* kTrackCuts = "Hybrid_LHC11h";
const char* kDatatype = "AOD";
const char* pass = "pass2";
TString kOrigClusName = "caloClusters";
TString kEMCALCellsName="emcalCells";
TString kInTracksName = "AODFilterTracks";
AliTrackContainer::SetDefTrackCutsPeriod("lhc11h");
}
