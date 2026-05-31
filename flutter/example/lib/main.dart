import 'package:flutter/material.dart';
import 'package:memlocal/memlocal.dart';

Future<void> main() async {
  await RustLib.init();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('memlocal quickstart')),
        body: Center(
          child: FutureBuilder<int>(
            future: Memlocal.openInMemory(dimensions: 1536)
                .then((m) => m.memoryCount()),
            builder: (context, snapshot) {
              if (snapshot.hasError) {
                return Text('Error: ${snapshot.error}');
              }
              if (!snapshot.hasData) {
                return const CircularProgressIndicator();
              }
              return Text(
                'Action: open in-memory engine + count\n'
                'Result: `${snapshot.data}`',
              );
            },
          ),
        ),
      ),
    );
  }
}
