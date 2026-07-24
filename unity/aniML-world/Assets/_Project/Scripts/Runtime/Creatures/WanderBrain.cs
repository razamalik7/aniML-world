using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// Placeholder brain: pick a nearby point, amble to it, idle, repeat.
    /// Exists so the sandbox has living motion before Malbers Animal
    /// Controller (and much later, RL policies) take over.
    /// </summary>
    public sealed class WanderBrain : ICreatureBrain
    {
        private readonly float _roamRadius;
        private readonly Vector2 _idleTimeRange;

        private CreatureController _controller;
        private Vector3 _home;
        private Vector3 _target;
        private float _idleUntil;
        private bool _idling = true;

        public WanderBrain(float roamRadius = 25f, Vector2? idleTimeRange = null)
        {
            _roamRadius = roamRadius;
            _idleTimeRange = idleTimeRange ?? new Vector2(2f, 6f);
        }

        public void Attach(CreatureController controller)
        {
            _controller = controller;
            _home = controller.transform.position;
            BeginIdle();
        }

        public CreatureIntent Think(float deltaTime)
        {
            var pos = _controller.transform.position;

            if (_idling)
            {
                if (Time.time < _idleUntil)
                    return CreatureIntent.Idle;

                PickNewTarget(pos);
            }

            var toTarget = _target - pos;
            toTarget.y = 0f;

            if (toTarget.sqrMagnitude < 1f)
            {
                BeginIdle();
                return CreatureIntent.Idle;
            }

            return new CreatureIntent
            {
                MoveDirection = toTarget.normalized,
                Speed01 = 0.4f, // amble; flee/chase states come with real brains
                Focus = null,
            };
        }

        private void BeginIdle()
        {
            _idling = true;
            _idleUntil = Time.time + Random.Range(_idleTimeRange.x, _idleTimeRange.y);
        }

        private void PickNewTarget(Vector3 from)
        {
            _idling = false;
            var offset = Random.insideUnitCircle * _roamRadius;
            _target = _home + new Vector3(offset.x, 0f, offset.y);
        }
    }
}
