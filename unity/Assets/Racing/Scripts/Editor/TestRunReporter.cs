using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace Racing.Editor
{
    /// <summary>
    /// Runs the Racing test assemblies from code (editor stays open) and writes a plain-text summary to
    /// outputs/test-results/&lt;Mode&gt;_summary.txt (paths.json test_results). Callbacks are re-registered after every
    /// domain reload.
    /// </summary>
    [InitializeOnLoad]
    public static class TestRunReporter
    {
        const string ModeKey = "Racing.TestRunReporter.Mode";
        public static string OutDir => RepoPaths.TestResults;

        static TestRunReporter()
        {
            ScriptableObject.CreateInstance<TestRunnerApi>().RegisterCallbacks(new Callbacks());
        }

        /// <param name="mode">"EditMode" or "PlayMode"</param>
        /// <param name="testNameFilter">optional full test names (exact)</param>
        public static void Run(string mode, params string[] testNameFilter)
        {
            Directory.CreateDirectory(OutDir);
            File.WriteAllText(StatusPath(mode), "running");
            SessionState.SetString(ModeKey, mode);
            var filter = new Filter
            {
                testMode = mode == "PlayMode" ? TestMode.PlayMode : TestMode.EditMode,
                assemblyNames = new[] { "Racing.Tests." + mode },
                testNames = testNameFilter != null && testNameFilter.Length > 0 ? testNameFilter : null
            };
            ScriptableObject.CreateInstance<TestRunnerApi>().Execute(new ExecutionSettings(filter));
        }

        static string StatusPath(string mode) => Path.Combine(OutDir, mode + "_status.txt");

        sealed class Callbacks : ICallbacks
        {
            public void RunStarted(ITestAdaptor testsToRun) { }
            public void TestStarted(ITestAdaptor test) { }
            public void TestFinished(ITestResultAdaptor result) { }

            public void RunFinished(ITestResultAdaptor result)
            {
                string mode = SessionState.GetString(ModeKey, "Unknown");
                var sb = new StringBuilder();
                var c = CultureInfo.InvariantCulture;
                sb.AppendFormat(c, "{0}: pass={1} fail={2} skip={3} inconclusive={4} duration={5:F1}s\n",
                    mode, result.PassCount, result.FailCount, result.SkipCount, result.InconclusiveCount, result.Duration);
                Append(result, sb, c);
                Directory.CreateDirectory(OutDir);
                File.WriteAllText(Path.Combine(OutDir, mode + "_summary.txt"), sb.ToString());
                File.WriteAllText(StatusPath(mode), "done");
            }

            static void Append(ITestResultAdaptor r, StringBuilder sb, CultureInfo c)
            {
                if (r.HasChildren)
                {
                    foreach (var child in r.Children) Append(child, sb, c);
                    return;
                }
                sb.AppendFormat(c, "[{0}] {1} ({2:F2}s)\n", r.TestStatus, r.FullName, r.Duration);
                if (!string.IsNullOrEmpty(r.Message)) sb.Append("    message: ").Append(r.Message.Trim()).Append('\n');
                if (!string.IsNullOrEmpty(r.Output)) sb.Append("    output: ").Append(r.Output.Trim().Replace("\n", "\n            ")).Append('\n');
                if (r.TestStatus == TestStatus.Failed && !string.IsNullOrEmpty(r.StackTrace))
                    sb.Append("    stack: ").Append(r.StackTrace.Trim().Replace("\n", "\n           ")).Append('\n');
            }
        }
    }
}
