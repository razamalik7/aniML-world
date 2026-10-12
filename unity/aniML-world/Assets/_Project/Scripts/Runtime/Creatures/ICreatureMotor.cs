namespace AniML.Creatures
{
    /// <summary>
    /// Motion layer of a creature: turns a <see cref="CreatureIntent"/> into
    /// actual movement + animation. Kept separate from the brain so either
    /// side can be replaced independently (e.g. Malbers Animal Controller
    /// motor with a wander brain, or the same motor driven by an RL brain).
    /// </summary>
    public interface ICreatureMotor
    {
        void Attach(CreatureController controller);
        void Apply(in CreatureIntent intent, float deltaTime);

        /// <summary>Current planar speed in m/s (for UI / debugging / brains).</summary>
        float CurrentSpeed { get; }
    }
}
