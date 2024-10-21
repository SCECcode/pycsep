from unittest import TestCase

import numpy

from csep.core.catalog_evaluations import resampled_magnitude_test
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
        self.assertEqual(result.test_distribution, [0])
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

    def tearDown(self):
        pass


class TestMLLTest(TestCase):
    def setUp(self):
        pass

    def unit_test1(self):
        pass

    def tearDown(self):
        pass


class TestMLLTestLight(TestCase):
    def setUp(self):
        pass

    def unit_test1(self):
        pass

    def tearDown(self):
        pass
