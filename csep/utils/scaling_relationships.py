class WellsAndCoppersmith:

    @staticmethod
    def mag_length_strike_slip(mag):
        """  Returns average rupture length for a given magnitude from Wells and Coppersmith
        Args:
            mag: magnitude

        Returns:
            rupture_legnth: average rupture length for a given magnitude in kilometers
        """
        return 10**(-3.22+0.69*mag)
