#!/usr/bin/env python
import xnat

XNAT_URL = "https://xnat.bmia.nl/"

with xnat.connect(XNAT_URL, user='username') as xnat_session:
    KNOAP_train_project = xnat_session.projects["KNOAP-train"]
    for xnat_subject in KNOAP_train_project.subjects.values():
        print("SUBJECT:",xnat_subject.label)
        xr_session = str(xnat_subject.label)+"_XR"
        xnat_experiment = KNOAP_train_project.subjects[xnat_subject.label].experiments[xr_session]
        print("XNAT experiment:",xnat_experiment)
        xnat_experiment.download_dir('./XR/')
        

        
 


