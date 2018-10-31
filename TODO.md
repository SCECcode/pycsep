### For Catalogs
1. add spatial binning to the catalogs
2. plotting routines for catalogs
 * event size given linear time
 * spatial distribution of events
 * cell-averaged, cell-median plots
 * look at pygmt for maps
3. magnitude frequency distribution
4. write documentation for catalog api
5. convert comcat eventlist to numpy/pandas format (need general parser for ucerf3 also)
6. write tests for catalog classes
7. implement load_catalog() function for single UCERF3Catalog

### For workflow manager
1. Implement runtime storage information using json files and/or SQLite database
2. Allow jobs to be submitted using slurm/pbs
3. Write methods to call functions as subprocess
4. General logging functions that will write logging information for project.
