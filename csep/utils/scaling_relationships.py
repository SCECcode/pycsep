class WellsAndCoppersmith:

    @staticmethod
    def mag_length_strike_slip(mag):
        """
        Returns the rupture vs. length
        # todo: implement uncertainty, right now im just using the mean value
        Args:
            mag: magnitude

        Returns:
            rupture_legnth: average rupture length for a given magnitude in kilometers
        """
        return 10**(-3.22+0.69*mag)

    @staticmethod
    def length_mag_strike_slip(length):
        pass