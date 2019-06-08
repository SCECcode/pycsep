"""
Contains the default configurations for the CSEP system.

TODO: Add user level configuration. Combine with default so user configuration takes precedence.
"""

# Add other machines here if running on a different computer
machine_config = {
    "hpc-usc": {
        "name": "hpc-usc",
        "url": "hpc.usc.edu",
        "hostname": "hpc-login",
        "email": "wsavran@usc.edu",
        "type": "slurm",
        "mpj_home": "/home/scec-00/kmilner/mpj-current",
        "partition": "scec",
        "max_cores": 20,
        "mem_per_node": 64
    },
    "csep-cert": {
        "name": "csep-cert",
        "url": "certification.usc.edu",
        "hostname": "csep2.localhost",
        "email": "wsavran@usc.edu",
        "type": "direct",
        "max_cores": 32,
        "mem_per_node": 192,
    },
    "default": {
        "name": "default",
        "url": "default",
        "hostname": "default",
        "email": "None",
        "type": "direct",
        "max_cores": 1,
        "mem_per_node": 2,
    }
}


forecast_config = {
    "ucerf3-etas":
        {
            "config_templ": "$ETAS_SIMS/ucerf3-defaults.json",
            "script_templ": "$ETAS_SIMS/template_files/hpc-usc.slurm",
            "env": {"ETAS_LAUNCHER": "/home/scec-00/wsavran/git/ucerf3-etas-launcher"}
        }
}