namespace AniML.Creatures
{
    /// <summary>
    /// Decision layer of a creature. Implementations decide *where to go and
    /// what to do*; they never move transforms directly.
    ///
    /// Planned implementations:
    ///  - WanderBrain (placeholder, ships now)
    ///  - AnimalControllerBrain (wraps Malbers AI once the package is imported)
    ///  - PolicyBrain (RL policy via Unity Inference Engine, much later)
    /// </summary>
    public interface ICreatureBrain
    {
        /// <summary>Called once when the brain is attached to its controller.</summary>
        void Attach(CreatureController controller);

        /// <summary>Produce the intent for this tick.</summary>
        CreatureIntent Think(float deltaTime);
    }
}
