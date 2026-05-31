import 'package:integration_test/integration_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:memlocal/memlocal.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async => await RustLib.init());
  test('Can call rust function', () async {
    final memlocal = await Memlocal.openInMemory(dimensions: 1536);
    expect(await memlocal.memoryCount(), 0);
  });
}
