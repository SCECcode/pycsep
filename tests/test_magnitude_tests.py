from unittest import TestCase

import numpy
import scipy.special

from csep.core.catalog_evaluations import resampled_magnitude_test, MLL_magnitude_test
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import CatalogForecast
from csep.core.regions import CartesianGrid2D


class TestResampledMagnitudeTest(TestCase):
    def setUp(self):
        self.mag_bins = [1.0, 2.0]
        self.region = CartesianGrid2D.from_origins(
            numpy.array([[0.0, 0.0]]), dh=1.0, magnitudes=self.mag_bins
        )

    @staticmethod
    def D_o(forecast, obs):
        """
        Independent implementation of D_o for the resampled M-test, just following the equations
         from Serafini et al., (2024)

        Args:
            forecast: A CatalogForecast object
            obs: A CSEPCatalog

        Returns:
            The observed statistic
        """
        d_o = 0
        N_u = numpy.sum([i.get_number_of_events() for i in forecast.catalogs])
        N_o = obs.get_number_of_events()
        for i in range(len(obs.region.magnitudes)):
            union_mag_counts = forecast.magnitude_counts()[i] * forecast.n_cat
            first_term = numpy.log10(union_mag_counts * N_o / N_u + 1)
            second_term = numpy.log10(obs.magnitude_counts()[i] + 1)
            d_o += (first_term - second_term) ** 2

        return d_o

    @staticmethod
    def D_j(forecast, obs, seed=None):
        """
        Independent implementation of D_j for the resampled M-test, just following the equations
         from Serafini et al., (2024)

        Args:
            forecast: A CatalogForecast object
            obs: A CSEPCatalog
            seed: random number seed value for the sampling of magnitudes
        Returns:
            The test statistic distribution
        """
        if seed:
            numpy.random.seed(seed)

        D = []
        N_u = numpy.sum([i.get_number_of_events() for i in forecast.catalogs])
        N_o = obs.get_number_of_events()
        mag_histogram = forecast.magnitude_counts()

        for n in range(forecast.n_cat):
            d_j = 0
            lambda_j = numpy.random.choice(
                obs.region.magnitudes, size=N_o, p=mag_histogram / numpy.sum(mag_histogram)
            )
            for i in range(len(obs.region.magnitudes)):
                union_mag_counts = forecast.magnitude_counts()[i] * forecast.n_cat
                first_term = numpy.log10(union_mag_counts * N_o / N_u + 1)
                counts = numpy.sum(lambda_j == obs.region.magnitudes[i])
                second_term = numpy.log10(counts + 1)
                d_j += (first_term - second_term) ** 2
            D.append(d_j)

        return D

    def test_no_region(self):
        synthetic_cats = [CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)])]
        forecast = CatalogForecast(catalogs=synthetic_cats)
        observation_cat = CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)])

        with self.assertRaises(AttributeError):
            resampled_magnitude_test(forecast, observation_cat)

    def test_single_cat_single_event(self):

        # Same magnitude as observation
        synthetic_cats = [CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)])]
        forecast = CatalogForecast(catalogs=synthetic_cats, region=self.region, n_cat=1)
        observation_cat = CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)], region=self.region)
        result = resampled_magnitude_test(forecast, observation_cat)
        self.assertEqual(result.test_distribution, [0])  # [0]
        self.assertEqual(result.observed_statistic, self.D_o(forecast, observation_cat))  # 0

        # Different magnitude as observation

        observation_cat = CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 2.0)], region=self.region)
        result = resampled_magnitude_test(forecast, observation_cat)
        self.assertEqual(result.test_distribution, [0])
        self.assertEqual(
            result.observed_statistic, self.D_o(forecast, observation_cat)
        )  # 0.18123811

    def test_single_cat_multiple_events(self):

        # Same MFD
        forecast = CatalogForecast(
            catalogs=[
                CSEPCatalog(
                    data=[
                        *(4 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),  # 4 events of this type
                        *(2 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),  # 2 events of this type
                    ]
                )
            ],
            region=self.region,
            n_cat=1,
        )

        observation_cat = CSEPCatalog(
            data=[
                *(2 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),  # 2 events of this type
                *(1 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),  # 1 event of this type
            ],
            region=self.region,
        )

        seed = 2  # Set seed so both methods sample the same events from the Union catalog.
        result = resampled_magnitude_test(forecast, observation_cat, seed=seed)
        self.assertEqual(result.observed_statistic, self.D_o(forecast, observation_cat))  # 0.
        self.assertEqual(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # [0.10622874]

        # Different MFD
        observation_cat = CSEPCatalog(
            data=[*(2 * [("0", 1, 0.0, 0.0, 0.0, 1.0)])],
            region=self.region,
        )
        seed = 3
        result = resampled_magnitude_test(forecast, observation_cat, seed=seed)
        self.assertEqual(
            result.observed_statistic, self.D_o(forecast, observation_cat)
        )  # 0.0611293829124204
        self.assertEqual(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # 0.010751542367500075

    def test_multiple_cat_multiple_events(self):

        # Same MFD
        forecast = CatalogForecast(
            catalogs=[
                CSEPCatalog(
                    data=[
                        *(2 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                        *(1 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
                    ]
                ),
                CSEPCatalog(
                    data=[
                        *(8 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                        *(4 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
                    ]
                ),
                CSEPCatalog(
                    data=[
                        *(12 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                        *(6 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
                    ]
                ),
            ],
            region=self.region,
            n_cat=3,
        )

        observation_cat = CSEPCatalog(
            data=[
                *(4 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                *(2 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
            ],
            region=self.region,
        )

        seed = 3  # Set seed so both methods sample the same events from the Union catalog.
        result = resampled_magnitude_test(forecast, observation_cat, seed=seed)

        self.assertEqual(result.observed_statistic, self.D_o(forecast, observation_cat))  # 0.
        self.assertEqual(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # [0.025001238526499825, 0.2489980945164454, 0.03727780124146953]

        # Different MFD
        observation_cat = CSEPCatalog(
            data=[*(3 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]), *(1 * [("0", 1, 0.0, 0.0, 0.0, 1.0)])],
            region=self.region,
        )
        seed = 3
        result = resampled_magnitude_test(forecast, observation_cat, seed=seed)
        self.assertAlmostEqual(
            result.observed_statistic, self.D_o(forecast, observation_cat)
        )  # 0.1535506203257525
        numpy.testing.assert_almost_equal(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # [0.005909847975937456, 0.019507668333914787, 0.1535506203257525]


class TestMLLTest(TestCase):
    def setUp(self):
        self.mag_bins = [1.0, 2.0]
        self.region = CartesianGrid2D.from_origins(
            numpy.array([[0.0, 0.0]]), dh=1.0, magnitudes=self.mag_bins
        )

    @staticmethod
    def D_o(forecast, catalog):

        def multinomial_likelihood(array):
            """
            Calculates the likelihood value. Only valid for small arrays and values.
            """
            total = numpy.sum(array)

            C = scipy.special.factorial(total)
            for i in array:
                C /= scipy.special.factorial(i)

            product = numpy.prod([(i / total) ** i for i in array])

            return C * product

        lambda_u = forecast.magnitude_counts() * forecast.n_cat
        omega = catalog.magnitude_counts()
        N_u = numpy.sum([i.get_number_of_events() for i in forecast.catalogs])
        N_o = catalog.get_number_of_events()

        likelihood_merged = multinomial_likelihood(lambda_u + N_u / N_o + omega + 1)
        likelihood_union = multinomial_likelihood(lambda_u + N_u / N_o)
        likelihood_cat = multinomial_likelihood(omega + 1)

        return 2 * numpy.log(likelihood_merged / likelihood_union / likelihood_cat)

    @staticmethod
    def D_j(forecast, catalog, seed=None):
        if seed:
            numpy.random.seed(seed)

        def multinomial_likelihood(array):
            """
            Calculates the likelihood value. Only valid for small arrays and values.
            """
            total = numpy.sum(array)

            C = scipy.special.factorial(total)
            for i in array:
                C /= scipy.special.factorial(i)

            product = numpy.prod([(i / total) ** i for i in array])

            return C * product

        D = []

        lambda_u = forecast.magnitude_counts() * forecast.n_cat
        N_u = numpy.sum([i.get_number_of_events() for i in forecast.catalogs])
        N_o = catalog.get_number_of_events()

        for n in range(forecast.n_cat):
            lambda_j = numpy.random.choice(
                catalog.region.magnitudes, size=N_o, p=lambda_u / numpy.sum(lambda_u)
            )
            counts = numpy.array([numpy.sum(lambda_j == i) for i in catalog.region.magnitudes])
            likelihood_merged = multinomial_likelihood(lambda_u + N_u / N_o + counts + 1)
            likelihood_union = multinomial_likelihood(lambda_u + N_u / N_o)
            likelihood_cat = multinomial_likelihood(counts + 1)
            D.append(2 * numpy.log(likelihood_merged / likelihood_union / likelihood_cat))

        return D

    def test_no_region(self):
        synthetic_cats = [CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)])]
        forecast = CatalogForecast(catalogs=synthetic_cats)
        observation_cat = CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)])

        with self.assertRaises(AttributeError):
            MLL_magnitude_test(forecast, observation_cat)

    def test_single_cat_single_event(self):

        # Same magnitudes as observation
        synthetic_cats = [CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)])]
        forecast = CatalogForecast(catalogs=synthetic_cats, region=self.region, n_cat=1)
        observation_cat = CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 1.0)], region=self.region)
        result = MLL_magnitude_test(forecast, observation_cat)

        # D_j = - 2 * log(  L(Lambda_U + N_u / N_j + Lambda_j + 1) /
        #                  [L(Lambda_U + N_u / N_j) * L(Lambda_j + 1)]
        #                )

        # First term: L(Lambda_U + N_u / N_j + Lambda_j + 1)
        #           Array: ( [1, 0] + 1/1 + [1, 0] + 1) =  ( [4, 2] )
        #               L ( [4, 2] ) = 6! / ( 4! * 2!) *   (4 / 6) ** 4 * (2 / 6) ** 2
        #                            =  5 * 6 / 2          *   2 ** 10 / 6 ** 6
        #                            = 0.3292181069958848

        # Second term: L(Lambda_U + N_u / N_j)
        #            Array: ([1, 0] + 1/1) =  ( [2, 1] )
        #               L ( [2, 1] ) = 3! / ( 2! * 1!) *   (2 / 3) ** 2 * (1 / 3) ** 1
        #                            =  3              *   4 / 9        *  1 / 3
        #                            =  0.4444444444444444

        # Third term:  L(Lambda_j + 1)
        #            Array: ([1, 0] + 1) =  ( [2, 1] )
        #               L ( [2, 1] ) =  0.4444444444444444

        # D_j = -2 log( 0.3292181069958848 / 0.4444444444444444 / 0.4444444444444444)
        #     = -1.0216512475319817
        self.assertAlmostEqual(result.test_distribution[0], 1.0216512475319817)

        # Different magnitude as observation
        observation_cat = CSEPCatalog(data=[("0", 1, 0.0, 0.0, 0.0, 3.0)], region=self.region)
        result = MLL_magnitude_test(forecast, observation_cat)
        # D_o = - 2 * log(  L(Lambda_U + N_u / N_o + Omega + 1) /
        #                  [L(Lambda_U + N_u / N_o) * L(Omega + 1)]
        #                )

        # First term: L(Lambda_U + N_u / N_o + Omega + 1)
        #           Array: ( [1, 0] + 1/1 + [0, 1] + 1) =  ( [3, 3] )
        #               L ( [3, 3] ) = 6! / ( 3! * 3!)          *   (3 / 6) ** 3 * (3 / 6) ** 3
        #                            =  4 * 5 * 6 / (1 * 2 * 3)   *   0.5 ** 6  * 0.5 ** 6
        #                            = 0.3125

        # Second term: L(Lambda_U + N_u / N_o)
        #            Array: ([1, 0] + 1/1) =  ( [2, 1] )
        #                            =  0.4444444444444444

        # Third term:  L(Omega + 1)
        #            Array: ([0, 1] + 1) =  ( [1, 2] )
        #               L ( [2, 1] ) =  0.4444444444444444

        # D_j = - 2 * log( 0.3125 / 0.4444444444444444 / 0.4444444444444444)
        #     = - 0.9174192452539534

        self.assertAlmostEqual(result.observed_statistic, 0.9174192452539534)
        # test own implementation
        self.assertAlmostEqual(result.observed_statistic, self.D_o(forecast, observation_cat))

    def test_single_cat_multiple_events(self):

        # Same MFD
        forecast = CatalogForecast(
            catalogs=[
                CSEPCatalog(
                    data=[
                        *(4 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),  # 4 events of this type
                        *(2 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),  # 2 events of this type
                    ]
                )
            ],
            region=self.region,
            n_cat=1,
        )

        observation_cat = CSEPCatalog(
            data=[
                *(2 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),  # 2 events of this type
                *(1 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),  # 1 event of this type
            ],
            region=self.region,
        )

        seed = 2  # Set seed so both methods sample the same events from the Union catalog.
        result = MLL_magnitude_test(forecast, observation_cat, seed=seed)
        self.assertAlmostEqual(
            result.observed_statistic, self.D_o(forecast, observation_cat)
        )  # 1.7370001360756202
        numpy.testing.assert_almost_equal(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # [1.4704757763861487]

        # Different MFD
        observation_cat = CSEPCatalog(
            data=[*(2 * [("0", 1, 0.0, 0.0, 0.0, 1.0)])],
            region=self.region,
        )
        seed = 3
        result = MLL_magnitude_test(forecast, observation_cat, seed=seed)
        self.assertAlmostEqual(
            result.observed_statistic, self.D_o(forecast, observation_cat)
        )  # 1.483977055813141
        numpy.testing.assert_almost_equal(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # [1.6728620032121357]

    def test_multiple_cat_multiple_events(self):

        # Same MFD
        forecast = CatalogForecast(
            catalogs=[
                CSEPCatalog(
                    data=[
                        *(2 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                        *(1 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
                    ]
                ),
                CSEPCatalog(
                    data=[
                        *(8 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                        *(4 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
                    ]
                ),
                CSEPCatalog(
                    data=[
                        *(12 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                        *(6 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
                    ]
                ),
            ],
            region=self.region,
            n_cat=3,
        )

        observation_cat = CSEPCatalog(
            data=[
                *(4 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]),
                *(2 * [("0", 1, 0.0, 0.0, 0.0, 2.0)]),
            ],
            region=self.region,
        )

        seed = 2  # Set seed so both methods sample the same events from the Union catalog.
        result = MLL_magnitude_test(forecast, observation_cat, seed=seed)
        self.assertAlmostEqual(
            result.observed_statistic, self.D_o(forecast, observation_cat)
        )  # 2.3691574300045595
        numpy.testing.assert_almost_equal(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # [1.752103594329519, 1.752103594329519, 2.3691574300045595]

        # Different MFD
        observation_cat = CSEPCatalog(
            data=[*(3 * [("0", 1, 0.0, 0.0, 0.0, 1.0)]), *(1 * [("0", 1, 0.0, 0.0, 0.0, 1.0)])],
            region=self.region,
        )
        seed = 3
        result = MLL_magnitude_test(forecast, observation_cat, seed=seed)
        self.assertAlmostEqual(
            result.observed_statistic, self.D_o(forecast, observation_cat)
        )  # 1.7348577364545044
        numpy.testing.assert_almost_equal(
            result.test_distribution, self.D_j(forecast, observation_cat, seed=seed)
        )  # [2.114537837794348, 2.202622612026193, 1.7348577364545044]
