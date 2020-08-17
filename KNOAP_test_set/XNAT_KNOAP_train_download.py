#!/usr/bin/env python
import xnat

XNAT_URL = "https://xnat.bmia.nl/"

with xnat.connect(XNAT_URL, user='jhirvasniemi') as xnat_session:
    KNOAP_train_project = xnat_session.projects["KNOAP-train"]
    KNOAP_train_project.download_dir('./')
    
