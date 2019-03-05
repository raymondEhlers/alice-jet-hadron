void handlers()
{
  {
    gROOT->LoadMacro(gSystem->ExpandPathName("$ALICE_ROOT/ANALYSIS/macros/train/AddAODHandler.C"));
    AliVEventHandler* handler = AddAODHandler();
  }
}
