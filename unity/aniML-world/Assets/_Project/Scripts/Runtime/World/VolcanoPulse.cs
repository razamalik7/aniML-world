using System.Collections.Generic;
using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// The volcano breathes: smoke and ember emission swell and ebb on a
    /// slow cycle, with occasional surges. Groundwork for the eruption
    /// event later. No-ops quietly on worlds without a volcano.
    /// </summary>
    public sealed class VolcanoPulse : MonoBehaviour
    {
        private readonly List<ParticleSystem> _vents = new();
        private readonly List<float> _baseRates = new();
        private float _surge;
        private float _nextSurgeIn = 30f;

        private void Start()
        {
            if (!Application.isPlaying)
                return;
            // The crater FX live under the wonders root, spawned from
            // prefabs whose names carry Smoke/Ember.
            var wonders = transform.Find("Wonders");
            if (wonders == null)
                return;
            foreach (var ps in wonders.GetComponentsInChildren<ParticleSystem>(true))
            {
                string n = ps.gameObject.name;
                if (!n.Contains("Smoke") && !n.Contains("Ember"))
                    continue;
                _vents.Add(ps);
                _baseRates.Add(ps.emission.rateOverTimeMultiplier);
            }
        }

        private void Update()
        {
            if (!Application.isPlaying || _vents.Count == 0)
                return;

            _nextSurgeIn -= Time.deltaTime;
            if (_nextSurgeIn <= 0f)
            {
                _surge = 1f;
                _nextSurgeIn = 45f + Random.value * 90f;
            }
            _surge = Mathf.MoveTowards(_surge, 0f, Time.deltaTime * 0.08f);

            float breath = 0.65f + 0.35f * Mathf.Sin(Time.time * 0.12f);
            float mul = breath * (1f + _surge * 2.2f);
            for (int i = 0; i < _vents.Count; i++)
            {
                if (_vents[i] == null)
                    continue;
                var em = _vents[i].emission;
                em.rateOverTimeMultiplier = _baseRates[i] * mul;
            }
        }
    }
}
