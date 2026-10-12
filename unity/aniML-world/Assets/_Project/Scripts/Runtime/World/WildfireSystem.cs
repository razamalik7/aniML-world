using System;
using System.Collections.Generic;
using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// Live wildfire: lightning (or a future god-tool) ignites a cell; fire
    /// spreads over a fuel/dryness grid with the wind (same FireRules the
    /// genesis burn used), chars the ground through a world-mapped burn
    /// mask, burns standing vegetation out of the scatter system, and dies
    /// against water, rock, rain, or exhausted fuel. The first disaster.
    /// </summary>
    [RequireComponent(typeof(ProceduralWorld))]
    public sealed class WildfireSystem : MonoBehaviour
    {
        [Serializable]
        public struct Slot
        {
            public string Key;
            public GameObject Prefab;
        }

        [SerializeField] private List<Slot> slots = new();
        [SerializeField] private float tickSeconds = 1.3f;
        [SerializeField] private int maxActiveFireFX = 28;

        private ProceduralWorld _world;
        private WeatherSystem _weather;
        private ScatterSystem _scatter;

        private int _res;
        private float _cell;
        private byte[,] _state;      // 0 clear, 1 burning, 2 burned
        private byte[,] _burnTicks;
        private readonly List<(int x, int z)> _burning = new();
        private readonly Dictionary<(int, int), GameObject> _fireFX = new();
        private Texture2D _burnMask;
        private float _timer;
        private float _lightningRoll;
        private System.Random _rng;
        private bool _maskDirty;

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
            _weather = GetComponent<WeatherSystem>();
            _scatter = GetComponent<ScatterSystem>();
            _world.EnsureSamplers();
            _rng = new System.Random(Environment.TickCount);

            _cell = 4f;
            _res = Mathf.CeilToInt(_world.Settings.WorldSize / _cell);
            _state = new byte[_res, _res];
            _burnTicks = new byte[_res, _res];

            _burnMask = new Texture2D(_res, _res, TextureFormat.R8, false, true)
            {
                wrapMode = TextureWrapMode.Clamp,
            };
            var fill = new byte[_res * _res];
            _burnMask.SetPixelData(fill, 0);
            _burnMask.Apply(false);
            Shader.SetGlobalTexture("_AniML_BurnMask", _burnMask);
            Shader.SetGlobalFloat("_AniML_WorldSize", _world.Settings.WorldSize);
            Shader.SetGlobalFloat("_AniML_BurnMaskOn", 1f);
        }

        private void OnDestroy()
        {
            Shader.SetGlobalFloat("_AniML_BurnMaskOn", 0f);
        }

        /// <summary>Start a fire at a world position (lightning now, god
        /// tools and eruptions later). Returns false if nothing can burn.</summary>
        public bool Ignite(Vector3 pos)
        {
            var (gx, gz) = CellOf(pos.x, pos.z);
            if (gx < 1 || gz < 1 || gx >= _res - 1 || gz >= _res - 1)
                return false;
            if (_state[gx, gz] != 0)
                return false;
            float wx = gx * _cell - _world.Settings.WorldSize * 0.5f;
            float wz = gz * _cell - _world.Settings.WorldSize * 0.5f;
            if (FireRules.Fuel(_world, wx, wz) < 0.06f)
                return false;
            _state[gx, gz] = 1;
            _burning.Add((gx, gz));
            SpawnFireFX(gx, gz);
            return true;
        }

        private void Update()
        {
            if (!Application.isPlaying || _world == null)
                return;

            // Lightning starts fires during rain spells.
            if (_weather != null && _weather.Current == WeatherSystem.Kind.Rain)
            {
                _lightningRoll += Time.deltaTime;
                if (_lightningRoll > 6f)
                {
                    _lightningRoll = 0f;
                    if (_rng.NextDouble() < 0.16)
                        Strike();
                }
            }

            if (_burning.Count == 0)
                return;
            _timer += Time.deltaTime;
            if (_timer < tickSeconds)
                return;
            _timer = 0f;
            Tick();
        }

        private void Strike()
        {
            float half = _world.Settings.WorldSize * 0.5f;
            for (int attempt = 0; attempt < 12; attempt++)
            {
                float x = ((float)_rng.NextDouble() * 2f - 1f) * half * 0.85f;
                float z = ((float)_rng.NextDouble() * 2f - 1f) * half * 0.85f;
                var pos = new Vector3(x, _world.SampleHeight(x, z), z);
                if (!Ignite(pos))
                    continue;
                var bolt = SpawnSlot("Lightning", pos + Vector3.up * 18f);
                if (bolt != null)
                    Destroy(bolt, 3f);
                break;
            }
        }

        private void Tick()
        {
            float half = _world.Settings.WorldSize * 0.5f;
            var wind = _world.Settings.WindDirection.normalized;
            float wetness = _weather != null ? _weather.Wetness : 0f;
            var nbors = new (int dx, int dz)[] { (1, 0), (-1, 0), (0, 1), (0, -1) };
            var ignited = new List<(int, int)>();

            for (int i = _burning.Count - 1; i >= 0; i--)
            {
                var (bx, bz) = _burning[i];

                // Heavy rain can douse a burning cell outright.
                if (wetness > 0.55f && _rng.NextDouble() < 0.4)
                {
                    BurnOut(bx, bz, i);
                    continue;
                }

                foreach (var (dx, dz) in nbors)
                {
                    int nx = bx + dx, nz = bz + dz;
                    if (nx < 1 || nz < 1 || nx >= _res - 1 || nz >= _res - 1
                        || _state[nx, nz] != 0)
                        continue;
                    float wx = nx * _cell - half;
                    float wz = nz * _cell - half;
                    float chance = FireRules.SpreadChance(
                        FireRules.Fuel(_world, wx, wz),
                        FireRules.Dryness(_world, wx, wz, wetness),
                        dx * wind.x + dz * wind.y);
                    if (_rng.NextDouble() < chance)
                    {
                        _state[nx, nz] = 1;
                        ignited.Add((nx, nz));
                    }
                }

                _burnTicks[bx, bz]++;
                if (_burnTicks[bx, bz] >= 3)
                    BurnOut(bx, bz, i);
            }

            foreach (var c in ignited)
            {
                _burning.Add(c);
                SpawnFireFX(c.Item1, c.Item2);
            }

            if (_maskDirty)
            {
                _burnMask.Apply(false);
                _maskDirty = false;
            }
        }

        private void BurnOut(int gx, int gz, int burningIndex)
        {
            _state[gx, gz] = 2;
            _burning.RemoveAt(burningIndex);
            float half = _world.Settings.WorldSize * 0.5f;
            float wx = gx * _cell - half;
            float wz = gz * _cell - half;

            // Char the ground and take the vegetation with it.
            _burnMask.SetPixel(gx, gz, new Color(0.85f, 0, 0, 1));
            _maskDirty = true;
            if (_scatter != null)
                _scatter.RemoveVegetationInRadius(
                    new Vector3(wx, _world.SampleHeight(wx, wz), wz), _cell * 0.8f);

            if (_fireFX.TryGetValue((gx, gz), out var fx) && fx != null)
                Destroy(fx, 1.5f);
            _fireFX.Remove((gx, gz));
        }

        private void SpawnFireFX(int gx, int gz)
        {
            if (_fireFX.Count >= maxActiveFireFX)
                return;
            float half = _world.Settings.WorldSize * 0.5f;
            float wx = gx * _cell - half;
            float wz = gz * _cell - half;
            var pos = new Vector3(wx, _world.SampleHeight(wx, wz), wz);
            var fire = SpawnSlot("Fire", pos);
            if (fire != null)
                _fireFX[(gx, gz)] = fire;
            var smoke = SpawnSlot("Smoke", pos + Vector3.up * 1.5f);
            if (smoke != null)
                Destroy(smoke, 14f);
        }

        private GameObject SpawnSlot(string key, Vector3 pos)
        {
            foreach (var s in slots)
            {
                if (s.Key != key || s.Prefab == null)
                    continue;
                var go = Instantiate(s.Prefab, pos, Quaternion.identity, transform);
                go.hideFlags = HideFlags.DontSave;
                return go;
            }
            return null;
        }

        private (int, int) CellOf(float x, float z)
        {
            float half = _world.Settings.WorldSize * 0.5f;
            return (Mathf.RoundToInt((x + half) / _cell), Mathf.RoundToInt((z + half) / _cell));
        }
    }
}
