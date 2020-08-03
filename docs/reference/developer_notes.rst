Developer Notes
===============

Last updated: 03 August 2020

Reproducibility Files
---------------------

Store information for reproducibility. This should include the following:
    1. version information (git hash of commit)
    2. forecast filename
    3. evaluation catalog (including necessary information to recreate the filtering properties); maybe just md5
    4. do we need calculation dates?

Evaluation Results
------------------

Each evaluation should return an evaluation result class that has an associated .plot() method. We should be able to
.plot() most everything that makes sense. How do we .plot() a catalog? Should we .plot() a region? For serialization, we
can identify the appropriate class as a string in the class state and use that to create the correct object on load.




