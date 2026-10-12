namespace AniML.World
{
    /// <summary>
    /// The biome vocabulary. Terrain color, ground textures, scatter species
    /// mixes, and (later) the ecological sim all key off this one enum —
    /// classification happens in ProceduralWorld.SampleBiome from smooth
    /// climate fields, so regions are contiguous by construction.
    /// </summary>
    public enum Biome
    {
        Ocean,
        Beach,
        Wetland,
        Meadow,
        Scrub,          // dry grassland / rain-shadow side
        ForestFloor,    // inside a forest mass
        AlpineMeadow,   // cold uplands below the rock line
        Rock,
        Snow,
        RiverChannel,
    }

    /// <summary>
    /// Rock character of a cliff wall — which pack's cliff meshes dress it.
    /// Chosen per wall from the climate fields so a run stays coherent:
    /// granite in the cold uplands, red rock in the dry, mossy stone in the
    /// temperate green.
    /// </summary>
    public enum CliffFamily
    {
        Granite,
        RedRock,
        Mossy,
    }
}
