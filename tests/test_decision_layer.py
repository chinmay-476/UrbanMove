import json
import os
import glob
import unittest

import app as urban_app


class DecisionLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_db_path = urban_app.DB_PATH
        cls.original_locality_path = urban_app.LOCALITY_PROFILES_PATH
        cls.test_db_dir = os.path.join(os.getcwd(), 'instance')
        cls.empty_profiles_path = os.path.join(os.getcwd(), 'tests', 'empty_profiles.csv')

    @classmethod
    def tearDownClass(cls):
        urban_app.DB_PATH = cls.original_db_path
        urban_app.LOCALITY_PROFILES_PATH = cls.original_locality_path
        for path in glob.glob(os.path.join(cls.test_db_dir, 'urbanmove_test_*.sqlite3')):
            try:
                os.remove(path)
            except OSError:
                pass
        if os.path.exists(cls.empty_profiles_path):
            os.remove(cls.empty_profiles_path)

    def setUp(self):
        urban_app.app.testing = True
        self.client = urban_app.app.test_client()

        self.test_db_path = os.path.join(self.test_db_dir, f'urbanmove_test_{self._testMethodName}.sqlite3')
        os.makedirs(os.path.dirname(self.test_db_path), exist_ok=True)
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass
        urban_app.DB_PATH = self.test_db_path
        urban_app.LOCALITY_PROFILES_PATH = self.original_locality_path
        urban_app._LOCALITY_PROFILE_CACHE = {'mtime': None, 'df': None}
        urban_app._TRUST_CACHE = {'mtime': None, 'map': None}
        urban_app.init_local_db()

    def test_cost_breakdown_defaults(self):
        response = self.client.post('/api/cost_breakdown', json={'rent': 25000})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['monthly_total'], 32000.0)
        self.assertEqual(payload['move_in_cash'], 115000.0)
        self.assertEqual(payload['twelve_month_total'], 467000.0)

    def test_cost_breakdown_overrides(self):
        response = self.client.post(
            '/api/cost_breakdown',
            json={
                'rent': 30000,
                'deposit_months': 1,
                'brokerage_months': 0,
                'maintenance': 1800,
                'utilities': 2200,
                'parking': 0,
                'moving_cost': 5000
            }
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['monthly_total'], 34000.0)
        self.assertEqual(payload['move_in_cash'], 69000.0)

    def test_personalized_localities_with_profiles(self):
        response = self.client.get('/api/personalized_localities?city=Mumbai&bhk=2&budget=60000&limit=3')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertGreaterEqual(payload['total_candidates'], 1)
        self.assertTrue(payload['recommendations'])
        first = payload['recommendations'][0]
        self.assertIn('match_score', first)
        self.assertIn('profile_source', first)
        self.assertIn('sample_trust_score', first)
        self.assertIn('sample_freshness_label', first)
        self.assertIn('sample_contact_type_label', first)
        self.assertNotEqual(str(first.get('profile_notes', '')).lower(), 'nan')

    def test_personalized_localities_without_profiles(self):
        with open(self.empty_profiles_path, 'w', encoding='utf-8') as handle:
            handle.write(','.join(urban_app.LOCALITY_PROFILE_COLUMNS) + '\n')

        urban_app.LOCALITY_PROFILES_PATH = self.empty_profiles_path
        urban_app._LOCALITY_PROFILE_CACHE = {'mtime': None, 'df': None}

        response = self.client.get('/api/personalized_localities?city=Delhi&bhk=2&budget=30000&limit=2')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['recommendations'])
        self.assertTrue(all(item['profile_source'] == 'fallback' for item in payload['recommendations']))

    def test_properties_include_trust_metadata(self):
        response = self.client.get('/api/properties?city=Mumbai&limit=2')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['properties'])
        first = payload['properties'][0]
        self.assertIn('trust_score', first)
        self.assertIn('trust_flags', first)
        self.assertIn('data_quality_flags', first)
        self.assertIn('freshness_label', first)
        self.assertIn('contact_type_label', first)

    def test_shortlist_saved_search_and_feedback_crud(self):
        shortlist_response = self.client.post(
            '/api/shortlist',
            json={'listing_id': 12, 'city': 'Mumbai', 'locality': 'Bandra West', 'rent': 55000, 'bhk': 2, 'size': 900}
        )
        self.assertEqual(shortlist_response.status_code, 200)
        shortlist_payload = shortlist_response.get_json()
        shortlist_id = shortlist_payload['item']['id']

        shortlist_list = self.client.get('/api/shortlist').get_json()
        self.assertEqual(shortlist_list['total'], 1)

        delete_shortlist = self.client.delete(f'/api/shortlist?id={shortlist_id}')
        self.assertEqual(delete_shortlist.status_code, 200)

        saved_search = self.client.post(
            '/api/saved_searches',
            json={
                'name': 'Mumbai 2BHK',
                'search_params': {
                    'prediction_inputs': {'city': 'Mumbai', 'bhk': 2},
                    'planner_preferences': {'cost_weight': 35}
                }
            }
        )
        self.assertEqual(saved_search.status_code, 200)
        saved_payload = saved_search.get_json()
        search_id = saved_payload['item']['id']

        saved_list = self.client.get('/api/saved_searches').get_json()
        self.assertEqual(saved_list['total'], 1)
        self.assertEqual(saved_list['items'][0]['search_params']['prediction_inputs']['city'], 'Mumbai')

        delete_search = self.client.delete(f'/api/saved_searches?id={search_id}')
        self.assertEqual(delete_search.status_code, 200)

        feedback_response = self.client.post(
            '/api/prediction_feedback',
            json={
                'input': {'city': 'Mumbai', 'bhk': 2},
                'predicted_rent': 42000,
                'actual_rent': 40000,
                'feedback_text': 'Negotiated lower after inspection'
            }
        )
        self.assertEqual(feedback_response.status_code, 200)
        feedback_payload = feedback_response.get_json()
        self.assertEqual(feedback_payload['feedback']['actual_rent'], 40000.0)


if __name__ == '__main__':
    unittest.main()
