using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// High-level movement/behavior intent produced by a brain each tick.
    /// Motors translate intent into actual motion (Animal Controller, simple
    /// steering, or eventually a physics policy). Keeping this a plain struct
    /// is what lets brains be swapped without touching the rest of the game.
    /// </summary>
    public struct CreatureIntent
    {
        /// <summary>Desired planar move direction in world space (XZ, normalized or zero).</summary>
        public Vector3 MoveDirection;

        /// <summary>Desired speed as a fraction of the creature's max speed [0..1].</summary>
        public float Speed01;

        /// <summary>Optional world-space point of interest (look at / interact with).</summary>
        public Vector3? Focus;

        public static CreatureIntent Idle => new CreatureIntent
        {
            MoveDirection = Vector3.zero,
            Speed01 = 0f,
            Focus = null,
        };
    }
}
