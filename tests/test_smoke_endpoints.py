import glob
import json
import os
import unittest

import app as urban_app


class SmokeEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_db_path = urban_app.DB_PATH
        cls.original_alerts_path = urban_app.ALERTS_PATH
        cls.test_db_dir = os.path.join(os.getcwd(), 'instance')
        cls.test_artifact_dir = os.path.join(os.getcwd(), 'tests', 'tmp_artifacts')
        os.makedirs(cls.test_db_dir, exist_ok=True)
        os.makedirs(cls.test_artifact_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        urban_app.DB_PATH = cls.original_db_path
        urban_app.ALERTS_PATH = cls.original_alerts_path

        for path in glob.glob(os.path.join(cls.test_db_dir, 'urbanmove_smoke_*.sqlite3')):
            try:
                os.remove(path)
            except OSError:
                pass

        for path in glob.glob(os.path.join(cls.test_artifact_dir, 'alerts_smoke_*.json')):
            try:
                os.remove(path)
            except OSError:
                pass

    def setUp(self):
        urban_app.app.testing = True
        self.client = urban_app.app.test_client()

        self.test_db_path = os.path.join(self.test_db_dir, f'urbanmove_smoke_{self._testMethodName}.sqlite3')
        self.test_alerts_path = os.path.join(self.test_artifact_dir, f'alerts_smoke_{self._testMethodName}.json')

        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        if os.path.exists(self.test_alerts_path):
            os.remove(self.test_alerts_path)

        urban_app.DB_PATH = self.test_db_path
        urban_app.ALERTS_PATH = self.test_alerts_path
        urban_app.init_local_db()

    def _first_non_empty(self, *values):
        for value in values:
            if value not in (None, ''):
                return value
        return None

    def _get_sample_properties(self, limit=3):
        response = self.client.get(f'/api/properties?limit={limit}')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['properties'])
        return payload['properties']

    def _build_prediction_payload(self, sample):
        city = self._first_non_empty(sample.get('City'), sample.get('city'), 'Mumbai')
        lat, lon = urban_app._get_city_coords(city)
        return {
            'city': city,
            'locality': self._first_non_empty(sample.get('Area Locality'), sample.get('locality'), ''),
            'bhk': int(self._first_non_empty(sample.get('BHK'), sample.get('bhk'), 2)),
            'size': float(self._first_non_empty(sample.get('Size'), sample.get('size'), 750)),
            'bathroom': int(self._first_non_empty(sample.get('Bathroom'), sample.get('bathroom'), 2)),
            'furnishing': self._first_non_empty(sample.get('Furnishing Status'), sample.get('furnishing'), 'Semi-Furnished'),
            'tenant': self._first_non_empty(sample.get('Tenant Preferred'), sample.get('tenant'), 'Bachelors/Family'),
            'bathroom_type': self._first_non_empty(sample.get('Bathroom_Type'), sample.get('bathroom_type'), 'Standard'),
            'area_type': self._first_non_empty(sample.get('Area Type'), sample.get('area_type'), 'Super Area'),
            'latitude': float(self._first_non_empty(sample.get('Latitude'), sample.get('latitude'), lat)),
            'longitude': float(self._first_non_empty(sample.get('Longitude'), sample.get('longitude'), lon))
        }

    def test_static_and_read_only_endpoints(self):
        properties = self._get_sample_properties(limit=3)
        sample = properties[0]
        sample_id = int(sample['id'])
        compare_ids = ','.join(str(item['id']) for item in properties)
        city = self._first_non_empty(sample.get('City'), 'Mumbai')
        bhk = int(self._first_non_empty(sample.get('BHK'), 2))
        locality = self._first_non_empty(sample.get('Area Locality'), '')
        rent = float(self._first_non_empty(sample.get('Rent'), 35000))

        home = self.client.get('/', buffered=True)
        self.assertEqual(home.status_code, 200)
        self.assertIn('UrbanMove', home.get_data(as_text=True))
        home.close()

        script = self.client.get('/script.js', buffered=True)
        self.assertEqual(script.status_code, 200)
        script.close()

        stats = self.client.get('/api/stats')
        self.assertEqual(stats.status_code, 200)
        self.assertIn('avg_rent', stats.get_json())

        cities = self.client.get('/api/cities')
        self.assertEqual(cities.status_code, 200)
        self.assertTrue(cities.get_json()['cities'])

        single_property = self.client.get(f'/api/properties/{sample_id}')
        self.assertEqual(single_property.status_code, 200)
        self.assertEqual(single_property.get_json()['id'], sample_id)

        map_data = self.client.get('/api/map_data')
        self.assertEqual(map_data.status_code, 200)
        self.assertIsInstance(map_data.get_json(), list)

        budget = self.client.get(f'/api/budget_advisor?budget={rent}&city={city}&bhk={bhk}')
        self.assertEqual(budget.status_code, 200)
        self.assertIn('recommendations', budget.get_json())

        market = self.client.get(f'/api/market_insights?predicted_rent={rent}&city={city}&bhk={bhk}')
        self.assertEqual(market.status_code, 200)
        self.assertIn('position_label', market.get_json())

        trends = self.client.get(
            f'/api/rent_trends?city={city}&bhk={bhk}&locality={locality}&months_history=6&months_forecast=3&predicted_rent={rent}'
        )
        self.assertEqual(trends.status_code, 200)
        self.assertIn('historical', trends.get_json())

        scorecard = self.client.get(f'/api/locality_scorecard?city={city}&bhk={bhk}')
        self.assertEqual(scorecard.status_code, 200)
        self.assertIn('scorecard', scorecard.get_json())

        commute = self.client.get(f'/api/commute_advisor?city={city}&bhk={bhk}&budget={rent}')
        self.assertEqual(commute.status_code, 200)
        self.assertIn('recommendations', commute.get_json())

        price_intel = self.client.get(f'/api/price_intelligence?city={city}&bhk={bhk}&limit=10')
        self.assertEqual(price_intel.status_code, 200)
        self.assertIn('summary', price_intel.get_json())

        compare = self.client.get(f'/api/compare_listings?ids={compare_ids}')
        self.assertEqual(compare.status_code, 200)
        self.assertGreaterEqual(len(compare.get_json()['comparisons']), 1)

        monitoring = self.client.get('/api/model_monitoring')
        self.assertEqual(monitoring.status_code, 200)
        self.assertTrue(monitoring.get_json()['model']['loaded'])

    def test_prediction_flow_and_mutating_endpoints(self):
        sample = self._get_sample_properties(limit=1)[0]
        payload = self._build_prediction_payload(sample)

        prediction = self.client.post('/api/predict', json=payload)
        self.assertEqual(prediction.status_code, 200)
        prediction_data = prediction.get_json()
        predicted_rent = float(prediction_data['predicted_rent'])
        self.assertGreater(predicted_rent, 0)

        cost_breakdown = self.client.post('/api/cost_breakdown', json={'rent': predicted_rent})
        self.assertEqual(cost_breakdown.status_code, 200)
        self.assertIn('monthly_total', cost_breakdown.get_json())

        localities = self.client.get(
            f"/api/personalized_localities?city={payload['city']}&bhk={payload['bhk']}&budget={predicted_rent}&limit=3"
        )
        self.assertEqual(localities.status_code, 200)
        self.assertIn('recommendations', localities.get_json())

        what_if = self.client.post('/api/what_if', json={'base': payload})
        self.assertEqual(what_if.status_code, 200)
        self.assertGreater(what_if.get_json()['scenario_count'], 0)

        similar = self.client.post('/api/similar_listings', json={**payload, 'limit': 3})
        self.assertEqual(similar.status_code, 200)
        self.assertIn('similar', similar.get_json())

        explanation = self.client.post('/api/explain_prediction', json=payload)
        self.assertEqual(explanation.status_code, 200)
        self.assertTrue(explanation.get_json()['top_impacts'])

        alert_create = self.client.post(
            '/api/alerts',
            json={
                'name': 'Smoke Test Alert',
                'city': payload['city'],
                'bhk': payload['bhk'],
                'budget': predicted_rent
            }
        )
        self.assertEqual(alert_create.status_code, 200)
        alert_id = alert_create.get_json()['alert']['id']

        alert_check = self.client.post('/api/alerts/check', json={'alert_id': alert_id})
        self.assertEqual(alert_check.status_code, 200)
        self.assertEqual(alert_check.get_json()['checked_alerts'], 1)

        alert_list = self.client.get('/api/alerts')
        self.assertEqual(alert_list.status_code, 200)
        self.assertEqual(len(alert_list.get_json()['alerts']), 1)

        delete_alert = self.client.delete(f'/api/alerts/{alert_id}')
        self.assertEqual(delete_alert.status_code, 200)
        self.assertEqual(delete_alert.get_json()['deleted_id'], alert_id)


if __name__ == '__main__':
    unittest.main()
