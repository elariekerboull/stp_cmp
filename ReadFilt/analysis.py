import awkward as ak
import numpy as np
from coffea import processor
import hist
import uproot

class StopAnalysisProcessorSMSLess(processor.ProcessorABC):

    def __init__(self, samples):

        self.samples = samples

        # Histograms you want
        self._accumulator = {

            "jet1_pt": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 0, 1000, name="pt_jet1", label="pt jet1 [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "jet1_eta": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -3, 3, name="eta_jet1", label="eta jet1", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "jet1_phi": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -np.pi, np.pi, name="phi_jet1", label="phi jet1", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "met": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 0, 1000, name="met", label="MET [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "HT": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 0, 1000, name="HT", label="HT [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "njet": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(10, 0, 10, name="n_jets", label="MET [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "pt_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 0, 100, name="pt_lepton", label="pT(l) [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "phi_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -np.pi, np.pi, name="phi_lepton", label="phi(l)", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "eta_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -3, 3, name="eta_lepton", label="eta(l)", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "charge_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(2, -1, 1, name="charge_lepton", label="charge(l)", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "RelIso_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(20, 0, 1.0, name="RelIso_lepton", label="RelIso(l)", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "AbsIso_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(20, 0, 20, name="AbsIso_lepton", label="AbsIso(l)", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "dxy_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(10, 0, 10, name="dxy_lepton", label="dxy(l) [mm]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "dz_lepton": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(10, 0, 10, name="dz_lepton", label="dz(l) [mm]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),
            
            "RelIso_vs_pt": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 0, 100, name="pt_lepton", label="pT(l) [GeV]", underflow=True, overflow=True),
                hist.axis.Regular(50, 0, 0.6, name="RelIso_lepton", label="RelIso(l)", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "AbsIso_vs_pt": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 0, 40, name="pt_lepton", label="pT(l) [GeV]", underflow=True, overflow=True),
                hist.axis.Regular(50, 0, 10, name="AbsIso_lepton", label="AbsIso(l)", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata["dataset"]
        process, sample, year = dataset.split("__")

        # --------------------
        # 1. Object Definition
        # --------------------
        jets = events.Jet
        muons = events.Muon
        electrons = events.Electron
        met = events.MET
        taus = events.Tau
        isotracks = events.IsoTrack
        CaloMet = events.CaloMET
        ChgedMet = events.ChsMET
        genpart = events.GenPart

        # --------------------
        # 2. Object Selection
        # --------------------
        good_jets = jets[
            (jets.pt > 30) &
            (abs(jets.eta) < 2.4) &
            (jets.jetId >= 2)
        ]

        #### tests bad region

        hcal_bad_region = (
        (good_jets.phi > -1.57) & (good_jets.phi < -0.87) &
        (good_jets.eta > -3.0) & (good_jets.eta < -1.3)
        )

        if year == "2018":
            good_jets = good_jets[~hcal_bad_region]

        def dphi(obj1_phi, obj2_phi):
            d = obj1_phi - obj2_phi
            return (d + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        dphi_iso = dphi(isotracks.phi, met.phi)
        mt_isotrack = (np.sqrt(2*isotracks.pt * met.pt * (1-np.cos(dphi_iso))))
        refined_isotracks = isotracks[
            (isotracks.pt > 10) &
            (abs(isotracks.eta) < 2.5) &
            (abs(isotracks.dz) < 0.1) &
            (abs(isotracks.dxy) < 0.2) &
            (isotracks.pfRelIso03_all < 0.1) &
            (mt_isotrack < 100)
        ]

        n_taus = ak.num(taus)
        n_muons = ak.num(muons)
        n_electrons = ak.num(electrons)

        # --------------------
        # 3. Event Selection
        # --------------------

        n_IsoTracks = ak.num(refined_isotracks)
        n_jets = ak.num(good_jets)
        good_jets_pad = ak.pad_none(good_jets, 5)   
        jet1_pt = ak.fill_none(good_jets_pad[:, 0].pt, np.nan)
        jet2_pt = ak.fill_none(good_jets_pad[:, 1].pt, np.nan)
        jet3_pt = ak.fill_none(good_jets_pad[:, 2].pt, np.nan)
        jet4_pt = ak.fill_none(good_jets_pad[:, 3].pt, np.nan)
        Ht = ak.sum(good_jets.pt, axis=1)
        jet1_eta = ak.fill_none(good_jets_pad[:, 0].eta, np.nan)
        jet1_phi = ak.fill_none(good_jets_pad[:, 0].phi, np.nan)
        """
        # Jets related
        jet2_eta = ak.fill_none(good_jets_pad[:, 1].eta, np.nan)
        jet3_eta = ak.fill_none(good_jets_pad[:, 2].eta, np.nan)
        jet4_eta = ak.fill_none(good_jets_pad[:, 3].eta, np.nan)
        jet2_phi = ak.fill_none(good_jets_pad[:, 1].phi, np.nan)
        jet3_phi = ak.fill_none(good_jets_pad[:, 2].phi, np.nan)
        jet4_phi = ak.fill_none(good_jets_pad[:, 3].phi, np.nan)
        jet5_phi = ak.fill_none(good_jets_pad[:, 4].phi, np.nan)
        dphi_jet1 = np.array(ak.to_numpy(dphi(jet1_phi, met.phi)))
        dphi_jet2 = np.array(ak.to_numpy(dphi(jet2_phi, met.phi)))
        dphi_jet3 = np.array(ak.to_numpy(dphi(jet3_phi, met.phi)))
        dphi_jet4 = np.array(ak.to_numpy(dphi(jet4_phi, met.phi)))
        dphi_jet5 = np.array(ak.to_numpy(dphi(jet5_phi, met.phi)))
        # mindphi = np.nanmin(np.stack([abs(dphi_jet1), abs(dphi_jet2), abs(dphi_jet3), abs(dphi_jet4)], axis=1), axis=1)

        jet2_missing = np.isnan(dphi_jet2)
        jet3_missing = np.isnan(dphi_jet3)
        jet4_missing = np.isnan(dphi_jet4)
        """
        met_cleaning_calo = (abs((met.pt/CaloMet.pt) - 1))
        met_cleaning_chged = dphi(ChgedMet.phi, met.phi)
        

        #Isolation condition
        good_iso_muon = (((muons.pt < 25) & ((muons.pfRelIso03_all*muons.pt) < 5)) | ((muons.pt >= 25) & (muons.pfRelIso03_all < 0.2)))
        good_iso_electron = (((electrons.pt < 25) & ((electrons.pfRelIso03_all*electrons.pt) < 5)) | ((electrons.pt >= 25) & (electrons.pfRelIso03_all < 0.2)))

        # Cutbased electrons and medium muonID with isolation condition
        good_electrons = (electrons.cutBased > 0) & good_iso_electron
        good_muons = (muons.mediumId) & good_iso_muon

        n_electrons = ak.sum(good_electrons, axis=1)
        n_muons = ak.sum(good_muons,axis=1)

        selection_base = (
            (np.array(jet1_pt) > 110) &
            (np.array(n_IsoTracks) == 0) &
            (np.array(n_jets) >= 1) &
            (np.array(met.pt) > 280) &
            (ak.sum(taus.idDeepTau2017v2p1VSe >= 1, axis=1) == 0) &
            ((n_muons + n_electrons) == 1) &            
            (np.array(Ht) > 200) &
            (met_cleaning_calo < 0.5) &
            (abs(met_cleaning_chged) < 2.0)
            # Only for non-leptonic case
            #(abs(dphi_jet1) > 0.5) &
            #(jet2_missing | (abs(dphi_jet2) > 0.5)) &
            #(jet3_missing | (abs(dphi_jet3) > 0.25)) &
            #(jet4_missing | (abs(dphi_jet4) > 0.25)) &
        )
        selection_electron = (n_electrons == 1) & (n_muons == 0)
        selection_muon = (n_muons == 1) & (n_electrons == 0)

        def convert_lepton(newlep):
            return ak.zip(
                {"pt": newlep.pt, "eta": newlep.eta, "phi": newlep.phi,
                "pfRelIso03_all": newlep.pfRelIso03_all, "charge": newlep.charge, "dxy": newlep.dxy, "dz": newlep.dz},
            )

        
        leptons = ak.concatenate([convert_lepton(muons[muons.mediumId & good_iso_muon]), convert_lepton(electrons[(electrons.cutBased > 0) & good_iso_electron])], axis=1)
        # 4. Weights
        # --------------------
        lumi = self.samples["luminosity"][year] * 1000  # convert from /fb to /pb
        if process == "4BD-500-490":
            xsec = events.xsec
        elif process == "4BD-500-420":
            xsec = events.xsec
        else:
            xsec = float(self.samples["MC"][process][sample][year]["xsec"])            

        Ngen = self.samples["MC"][process][sample][year]["nEvents"]

        if "genWeight" in events.fields:
            genw = events.genWeight
        else:
            print(f"genWeight not found in {dataset}, using genw=1")
            genw = np.ones(len(events))
        
        weights = genw * lumi * xsec / Ngen
        has_jet2 = n_jets >= 2
        has_jet3 = n_jets >= 3
        has_jet4 = n_jets >= 4

        # --------------------
        # 5. Fill Histograms
        # --------------------
        self._accumulator["jet1_pt"].fill(
            process=process,
            year=year,
            pt_jet1=jet1_pt[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["jet1_eta"].fill(
            process=process,
            year=year,
            eta_jet1=jet1_eta[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["jet1_phi"].fill(
            process=process,
            year=year,
            phi_jet1=jet1_phi[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["met"].fill(
            process=process,
            year=year,
            met=met.pt[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["HT"].fill(
            process=process,
            year=year,
            HT=Ht[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["njet"].fill(
            process=process,
            year=year,
            n_jets=n_jets[selection_base],
            weight=weights[selection_base]
        )
        lepton = ak.firsts(leptons[selection_base])
        self._accumulator["pt_lepton"].fill(
            process=process,
            year=year,
            pt_lepton=lepton.pt,
            weight=weights[selection_base]
        )

        self._accumulator["phi_lepton"].fill(
            process=process,
            year=year,
            phi_lepton=lepton.phi,
            weight=weights[selection_base]
        )

        self._accumulator["eta_lepton"].fill(
            process=process,
            year=year,
            eta_lepton=lepton.eta,
            weight=weights[selection_base]
        )

        self._accumulator["charge_lepton"].fill(
            process=process,
            year=year,
            charge_lepton=lepton.charge,
            weight=weights[selection_base]
        )

        # Think - plot muon IDs all together?
        #self._accumulator["muon_looseID"].fill(
        #    process=process,
        #    year=year,
        #    njet=events.Muon.looseId,
        #    weight=weights
        #)

        #self._accumulator["muon_mediumID"].fill(
        #    process=process,
        #    year=year,
        #    njet=events.Muon.mediumId,
        #    weight=weights
        #)

        #self._accumulator["muon_tightID"].fill(
        #    process=process,
        #    year=year,
        #    njet=events.Muon.tightId,
        #    weight=weights
        #)

        self._accumulator["RelIso_lepton"].fill(
            process=process,
            year=year,
            RelIso_lepton=lepton.pfRelIso03_all,
            weight=weights[selection_base]
        )

        self._accumulator["AbsIso_lepton"].fill(
            process=process,
            year=year,
            AbsIso_lepton=(lepton.pfRelIso03_all*lepton.pt),
            weight=weights[selection_base]
        )

        self._accumulator["dxy_lepton"].fill(
            process=process,
            year=year,
            dxy_lepton=lepton.dxy,
            weight=weights[selection_base]
        )

        self._accumulator["dz_lepton"].fill(
            process=process,
            year=year,
            dz_lepton=lepton.dz,
            weight=weights[selection_base]
        )

        #self._accumulator["electron_cutbased"].fill(
        #    process=process,
        #    year=year,
        #    njet=events.Electron.cutBased,
        #    weight=weights
        #)

        self._accumulator["RelIso_vs_pt"].fill(
            process=process,
            year=year,
            pt_lepton=lepton.pt,
            RelIso_lepton=lepton.pfRelIso03_all,
            weight=weights[selection_base]
        )

        self._accumulator["AbsIso_vs_pt"].fill(
            process=process,
            year=year,
            pt_lepton=lepton.pt,
            AbsIso_lepton=(lepton.pfRelIso03_all * lepton.pt),
            weight=weights[selection_base]
        )

        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator
