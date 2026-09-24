using System;
using System.IO;
using UnityEngine;

namespace Racing.Editor
{
    /// <summary>
    /// Repository paths shared with Python (racing_rl.paths). paths.json at the repo root is the single source for every
    /// cross-language path (player builds, benchmark outputs, test results, the Python track catalog); the root is the
    /// first directory above the Unity project that contains it.
    /// </summary>
    public static class RepoPaths
    {
        public const string FileName = "paths.json";

        [Serializable]
        sealed class Config
        {
            public string unity_project;
            public string bridge_exe;
            public string mla_exe;
            public string benchmarks;
            public string runs;
            public string test_results;
            public string track_catalog;
        }

        static string _root;
        static Config _config;

        public static string Root => _root ??= FindRoot();
        static Config Cfg => _config ??= JsonUtility.FromJson<Config>(File.ReadAllText(Path.Combine(Root, FileName)));

        public static string BridgeExe => Resolve(Cfg.bridge_exe);
        public static string MlaExe => Resolve(Cfg.mla_exe);
        public static string Benchmarks => Resolve(Cfg.benchmarks);
        public static string Runs => Resolve(Cfg.runs);
        public static string TestResults => Resolve(Cfg.test_results);
        public static string TrackCatalog => Resolve(Cfg.track_catalog);
        public static string TrackCatalogRelPath => Cfg.track_catalog;

        /// <summary>Absolute path of a repo-relative path; an absolute path passes through unchanged.</summary>
        public static string Resolve(string repoRelative) => Path.GetFullPath(Path.Combine(Root, repoRelative));

        static string FindRoot()
        {
            for (var dir = new DirectoryInfo(Path.GetDirectoryName(Application.dataPath)); dir != null; dir = dir.Parent)
                if (File.Exists(Path.Combine(dir.FullName, FileName))) return dir.FullName;
            throw new FileNotFoundException(FileName + " not found above the Unity project", Application.dataPath);
        }
    }
}
