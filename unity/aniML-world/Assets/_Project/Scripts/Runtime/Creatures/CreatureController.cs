using AniML.World;
using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// The one component every creature carries. Owns identity (definition),
    /// a brain (decisions) and a motor (motion), and ticks them. Gameplay
    /// systems (spawning, selection, population UI, events) talk to this —
    /// never to a concrete brain or motor.
    /// </summary>
    public sealed class CreatureController : MonoBehaviour
    {
        [SerializeField] private CreatureDefinition definition;

        private ICreatureBrain _brain;
        private ICreatureMotor _motor;
        private ProceduralWorld _world;

        public CreatureDefinition Definition => definition;
        public CreatureIntent CurrentIntent { get; private set; } = CreatureIntent.Idle;
        public ICreatureBrain Brain => _brain;
        public ICreatureMotor Motor => _motor;
        public ProceduralWorld World => _world != null ? _world : (_world = FindAnyObjectByType<ProceduralWorld>());

        private void Awake()
        {
            if (_motor == null && TryGetComponent<ICreatureMotor>(out var attachedMotor))
                SetMotor(attachedMotor);
            if (_brain == null && TryGetComponent<ICreatureBrain>(out var attachedBrain))
                SetBrain(attachedBrain);
        }

        public void Initialize(CreatureDefinition def, ICreatureBrain brain, ICreatureMotor motor)
        {
            if (def != null)
                definition = def;
            SetMotor(motor);
            SetBrain(brain);
        }

        /// <summary>Swap the decision layer at runtime (e.g. wander → policy).</summary>
        public void SetBrain(ICreatureBrain brain)
        {
            _brain = brain;
            _brain?.Attach(this);
        }

        public void SetMotor(ICreatureMotor motor)
        {
            _motor = motor;
            _motor?.Attach(this);
        }

        private void Update()
        {
            if (_brain == null || _motor == null)
                return;

            CurrentIntent = _brain.Think(Time.deltaTime);
            _motor.Apply(CurrentIntent, Time.deltaTime);
        }
    }
}
