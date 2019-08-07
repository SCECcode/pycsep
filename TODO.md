**news tests / features**
* magnitude vs. time for catalog with same number of events
* go over likelihood test and spatial test

**optimizations**
* should only try to read data once (mem management, need to write classes)
* try and refactor numpy part and use numba jit to optimize those calls (after tests confirmed)
* refactor evaluations into classes that allow updating by catalog (only read/loop through catalog once)

**producing results**
* get feedback on figures from csep call today.
* setup automated evaluations, and reproducibility.

**repo love**
* tests for spatial region.
* tests testing for catalogs.
* tests tests for evaluations.
* write documentation and tutorial files. 
* look to merge into master and have first minor release when the above are complete.