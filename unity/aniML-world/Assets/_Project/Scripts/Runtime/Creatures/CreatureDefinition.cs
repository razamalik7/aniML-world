using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// Static per-species data. One asset per species (Deer, Wolf, ...).
    /// Ecology fields (diet, energy, reproduction) will grow here as the
    /// simulation layers arrive; keep gameplay data out of prefabs so
    /// species can be tuned without touching scenes.
    /// </summary>
    [CreateAssetMenu(menuName = "AniML/Creature Definition", fileName = "NewCreature")]
    public sealed class CreatureDefinition : ScriptableObject
    {
        [Header("Identity")]
        public string SpeciesId = "deer";
        public string DisplayName = "Deer";

        [Header("Locomotion")]
        [Tooltip("Top speed in m/s at Speed01 = 1.")]
        public float MaxSpeed = 6f;
        [Tooltip("Degrees per second the creature can turn.")]
        public float TurnRate = 240f;

        [Header("Presentation")]
        public GameObject Prefab;
    }
}
