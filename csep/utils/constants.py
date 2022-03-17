# Time Constants
import numpy

SECONDS_PER_ASTRONOMICAL_YEAR = 31557600
SECONDS_PER_DAY = 60*60*24
SECONDS_PER_HOUR = 60*60
SECONDS_PER_WEEK = SECONDS_PER_DAY*7
SECONDS_PER_MONTH = SECONDS_PER_WEEK*4
DAYS_PER_ASTRONOMICAL_YEAR = 365.25

MW_5_EQS_PER_YEAR = 10

earth_radius_km = 6371.

# Magnitude Bins
min_mw = 2.5
max_mw = 8.95
dmw = 0.1
CSEP_MW_BINS = numpy.array([  2.5,   2.6,   2.7,   2.8,   2.9,   3. ,   3.1,   3.2,   3.3,
                              3.4,   3.5,   3.6,   3.7,   3.8,   3.9,   4. ,   4.1,   4.2,
                              4.3,   4.4,   4.5,   4.6,   4.7,   4.8,   4.9,   5. ,   5.1,
                              5.2,   5.3,   5.4,   5.5,   5.6,   5.7,   5.8,   5.9,   6. ,
                              6.1,   6.2,   6.3,   6.4,   6.5,   6.6,   6.7,   6.8,   6.9,
                              7. ,   7.1,   7.2,   7.3,   7.4,   7.5,   7.6,   7.7,   7.8,
                              7.9,   8. ,   8.1,   8.2,   8.3,   8.4,   8.5,   8.6,   8.7,
                              8.8,   8.9,   9. ,   9.1,   9.2,   9.3,   9.4,   9.5,   9.6,
                              9.7,   9.8,   9.9,  10. ])

