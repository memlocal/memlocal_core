import 'package:flutter/material.dart';
import 'package:memlocal/memlocal.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  runApp(const SmokeApp());
}

class SmokeApp extends StatefulWidget {
  const SmokeApp({super.key});
  @override
  State<SmokeApp> createState() => _SmokeAppState();
}

class _SmokeAppState extends State<SmokeApp> {
  String _status = 'opening engine…';

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    try {
      final mem = await Memlocal.openInMemory(dimensions: 1536);
      final count = await mem.memoryCount();
      final doubled =
          await callDartClosure(value: 21, callback: (v) async => v * 2);
      setState(() => _status =
          'OK — engine open, memoryCount=$count, callback(21)=$doubled');
    } catch (e) {
      setState(() => _status = 'FAILED: $e');
    }
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('memlocal smoke test')),
          body: Center(child: Text(_status, key: const Key('status'))),
        ),
      );
}
