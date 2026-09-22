using Racing.Core;
using UnityEngine;

namespace Racing.Tests
{
    static class TestUtil
    {
        public static T Load<T>(string path, System.Func<T> fallback) where T : ScriptableObject
        {
#if UNITY_EDITOR
            var asset = UnityEditor.AssetDatabase.LoadAssetAtPath<T>(path);
            if (asset != null) return asset;
#endif
            return fallback();
        }

        public static SimConfig Sim => Load("Assets/Racing/Config/SimConfig.asset", SimConfig.CreateDefault);
        public static VehicleConfig Vehicle => Load("Assets/Racing/Config/VehicleConfig.asset", VehicleConfig.CreateDefault);
        public static TrackDefinition Track => Load("Assets/Racing/Config/TrackDefinition_A.asset", TrackDefinition.CreateDefault);

        public static RaceEnvironment CreateEnv(int agents, long seed, StartMode mode) =>
            RaceEnvironment.Create(Track, Vehicle, Sim, agents, seed, mode);

        public static void Destroy(Object o)
        {
            if (o != null) Object.DestroyImmediate(o is Component c ? c.gameObject : o);
            Physics.SyncTransforms();
        }

        /// <summary>Flat test ground (Road layer) without a track.</summary>
        public static GameObject CreateGround(float size = 4000f)
        {
            var g = GameObject.CreatePrimitive(PrimitiveType.Cube);
            g.name = "TestGround";
            g.layer = RacingLayers.Road;
            g.transform.position = new Vector3(0f, -0.5f, 0f);
            g.transform.localScale = new Vector3(size, 1f, size);
            return g;
        }

        public static void UseScriptPhysics(SimConfig sim)
        {
            CultureBootstrap.Apply();
            Physics.simulationMode = SimulationMode.Script;
            Time.fixedDeltaTime = sim.fixedDeltaTime;
            Physics.IgnoreLayerCollision(RacingLayers.Car, RacingLayers.Car, true);
        }
    }
}
