using System;
using System.Collections.Generic;
using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// Weather spells: Clear / Rain / Snow / Fog rolling on a timer. The
    /// particle art is Synty's (PNB_Core); the logic is ours. Emitters follow
    /// the main camera so precipitation exists wherever you look. Snow only
    /// falls in worlds cold enough to carry it; deserts mostly stay dry.
    /// </summary>
    public sealed class WeatherSystem : MonoBehaviour
    {
        public enum Kind { Clear, Rain, Snow, Fog }

        [Serializable]
        public struct Slot
        {
            public string Key;
            public GameObject Prefab;
        }

        [SerializeField] private List<Slot> slots = new();
        [SerializeField] private float minSpellSeconds = 50f;
        [SerializeField] private float maxSpellSeconds = 140f;

        public Kind Current { get; private set; } = Kind.Clear;

        private ProceduralWorld _world;
        private GameObject _activeFX;
        private float _timer;
        private float _fogTarget;
        private System.Random _rng;

        public void SetSlot(string key, GameObject prefab)
        {
            for (int i = 0; i < slots.Count; i++)
            {
                if (slots[i].Key == key)
                {
                    slots[i] = new Slot { Key = key, Prefab = prefab };
                    return;
                }
            }
            slots.Add(new Slot { Key = key, Prefab = prefab });
        }

        private void Start()
        {
            if (!Application.isPlaying)
                return;
            _world = GetComponent<ProceduralWorld>();
            _rng = new System.Random(Environment.TickCount);
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Exponential;
            RenderSettings.fogDensity = 0.0005f;
            _fogTarget = 0.0005f;
            _timer = Roll();
        }

        private void Update()
        {
            if (!Application.isPlaying)
                return;

            _timer -= Time.deltaTime;
            if (_timer <= 0f)
                Force(PickNext(), Roll());

            // Ease fog toward the current spell's density.
            RenderSettings.fogDensity = Mathf.Lerp(
                RenderSettings.fogDensity, _fogTarget, Time.deltaTime * 0.35f);

            // Precipitation follows the camera.
            if (_activeFX != null && Camera.main != null)
            {
                var c = Camera.main.transform.position;
                _activeFX.transform.position = new Vector3(c.x, c.y + 18f, c.z);
            }
        }

        /// <summary>Switch weather immediately (also used by events later —
        /// a storm the World Director summons is just Force(Rain).)</summary>
        public void Force(Kind kind, float seconds)
        {
            Current = kind;
            _timer = seconds;

            if (_activeFX != null)
                Destroy(_activeFX);
            _activeFX = null;

            switch (kind)
            {
                case Kind.Rain:
                    _activeFX = SpawnFX("Rain");
                    _fogTarget = 0.0022f;
                    break;
                case Kind.Snow:
                    _activeFX = SpawnFX("Snow");
                    _fogTarget = 0.0028f;
                    break;
                case Kind.Fog:
                    _fogTarget = 0.010f;
                    break;
                default:
                    _fogTarget = 0.0005f;
                    break;
            }
        }

        private Kind PickNext()
        {
            // Cold worlds snow, wet worlds rain, dry worlds mostly shrug.
            bool cold = _world != null
                && _world.SampleTemperature01(0f, 0f) < 0.42f;
            bool dry = _world != null
                && _world.SampleMoisture(0f, 0f) < 0.3f;

            double r = _rng.NextDouble();
            if (Current != Kind.Clear)
                return Kind.Clear;   // spells always break back to clear
            if (dry)
                return r < 0.75 ? Kind.Clear : (r < 0.9 ? Kind.Fog : Kind.Rain);
            if (cold)
                return r < 0.45 ? Kind.Clear : (r < 0.75 ? Kind.Snow : Kind.Fog);
            return r < 0.45 ? Kind.Clear : (r < 0.8 ? Kind.Rain : Kind.Fog);
        }

        private float Roll() =>
            Mathf.Lerp(minSpellSeconds, maxSpellSeconds, (float)_rng.NextDouble());

        private GameObject SpawnFX(string key)
        {
            foreach (var s in slots)
            {
                if (s.Key != key || s.Prefab == null)
                    continue;
                var go = Instantiate(s.Prefab, transform);
                go.hideFlags = HideFlags.DontSave;
                return go;
            }
            return null;
        }
    }
}
