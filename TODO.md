**news tests / features**
* total event rate distribution
* inter-event time/distance distribution
* depth distribution test
* b-value distribution test
* magnitude vs. time for catalog with same number of events
* go over likelihood test and spatial test

**optimizations**
* should only try to read data once (mem management)
* try and refactor numpy part and use numba jit to optimize those calls (after tests confirmed)

**producing results**
* some figures need modified a bit
* setup automated evaluations, and reproducibility.

**repo love**
* tests for spatial region.
* tests testing for catalogs.
* tests tests for evaluations.
* write documentation and tutorial files. 
* look to merge into master and have first minor release when the above are complete.