#!/usr/bin/perl

# JMA REDC.DECK fomat ¤«¤é ¿Ì¸»¾ðÊó¤ò¼è¤ê½Ð¤¹¥¹¥¯¥ê¥×¥È
# 98.10.22 tsuru
# updated to create CSV format hypocenter file. -tb 2020-02-13 (tb@gfz.pm)

# printf("year;month;day;hour;minute;second;longitude;latitude;depth;magnitude\n");

# running like this:
# time for file in $(ls ../../deckfiles/??????*.deck.Z | sort); do uncompress -c $file; done | time ./deck2csv.pl > ../../JMA-hypocenter-timestamp.csv


printf("timestamp;longitude;latitude;depth;magnitude\n");

while(<>){
        chop;
        if ( /^J/ ) {
                $cmag = substr($_, 52, 2);
                s/ /0/g;
                $yr = substr($_,  1, 4);
                $mo = substr($_,  5, 2);
                $dy = substr($_,  7, 2);
                $hr = substr($_,  9, 2);
                $mi = substr($_, 11, 2);
                
                # $sc = substr($_, 13, 4); $sc = $sc/100.0;
                $sc = substr($_, 13, 2);
                $sf = substr($_, 15, 2); $sf = $sf * 10000;

                # $dt = substr($_, 17, 4); $dt = $dt/100.0;
                $latd = substr($_, 21, 3);
                $latm = substr($_, 24, 4);
                $lat = $latd + $latm/(100.0*60.0);
                $elat = substr($_, 28, 4); $elat = $elat/(100.0*60.0);
                $lond = substr($_, 32, 4);
                $lonm = substr($_, 36, 4);
                $lon = $lond + $lonm/(100.0*60.0);
                $elon = substr($_, 40, 4); $elon = $elon/(100.0*60.0);
                $dep = substr($_, 44, 5); $dep = $dep/100.0;
                $edep = substr($_, 49, 3);
                if ( $edep eq '000' ) {
                        $edep = 1.0;
                } else {
                        $edep = $edep/100.0;
                }
#               $cmag = substr($_, 52, 2);
                $mflag = substr($_, 52, 1);
                $m = substr($_, 53, 1);
                if ( $mflag eq 'A') { $cmag = '-1' . $m; }
                if ( $mflag eq 'B') { $cmag = '-2' . $m; }
                if ( $mflag eq 'C') { $cmag = '-3' . $m; }
                $mag = $cmag; $mag = $mag/10.0;
                if ( $cmag ne '  ' ) {
                  printf("%04d-%02d-%02dT%02d:%02d:%02d.%06d+0900;", $yr, $mo, $dy, $hr, $mi, $sc, $sf);
                  printf("%.4f;%.4f;%.2f;%.3f\n", $lon, $lat, $dep, $mag);
                  # printf("%04d;%02d;%02d;%02d;%02d;%.2f;", $yr, $mo, $dy, $hr, $mi, $sc);
                  # printf("%.4f;%.4f;%.2f;%.3f\n", $lon, $lat, $dep, $mag);
                }
        }
}


# J2013010100033169 015 372521 029 1412406 076 279510616V   511   2 69E OFF FUKUSHIMA PREF     30K
