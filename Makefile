# libchaff Makefile
PREFIX ?= /usr/local
DESTDIR ?=

CC ?= gcc
CFLAGS ?= -Wall -Wextra -Werror -O2 -Iinclude
LDLIBS ?= -lxxhash

WASM_CC ?= clang
WASM_CFLAGS ?= --target=wasm32-wasi -ffreestanding -flto -O3 -D__wasm__ -fvisibility=hidden -I./include -I./npm/include -I../libstele/include -I../mk/include -I/usr/include
WASM_LDFLAGS ?= -mexec-model=reactor -Wl,--gc-sections -Wl,--no-entry -Wl,--allow-undefined \
	-Wl,--import-memory -Wl,-z,stack-size=32768 -Wl,--global-base=1638400 \
	-Wl,--wrap=malloc -Wl,--wrap=free -Wl,--wrap=calloc -Wl,--wrap=realloc \
	-Wl,--export=noise_wasm_point_1d -Wl,--export=noise_wasm_point_2d -Wl,--export=noise_wasm_point_3d -Wl,--export=noise_wasm_point_4d -Wl,--export=noise_wasm_point_n -Wl,--export=noise_wasm_octaves

SO_CFLAGS ?= -mgeneral-regs-only

TEST_BINS := bin/test_noise bin/test_chaff bin/test_symbols bin/test_glsl bin/test_chaff_multiview
BENCH_BINS := bin/bench_noise

all: $(TEST_BINS) $(BENCH_BINS) lib/libchaff.so wasm

test: $(TEST_BINS)
	./bin/test_noise
	./bin/test_chaff
	./bin/test_symbols
	./bin/test_glsl
	./bin/test_chaff_multiview

bench: $(BENCH_BINS)
	./bin/bench_noise

bin/bench_noise: tests/bench_noise.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ tests/bench_noise.c $(LDLIBS) -lm

bin/test_noise: tests/test_noise.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ tests/test_noise.c $(LDLIBS) -lm

bin/test_chaff: tests/test_chaff.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ tests/test_chaff.c $(LDLIBS) -lm

bin/test_symbols: tests/test_symbols.c src/libchaff.c include/ttypt/noise.h include/ttypt/chaff.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ tests/test_symbols.c src/libchaff.c $(LDLIBS) -lm

bin/test_glsl: tests/test_glsl.c include/ttypt/chaff_glsl.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ tests/test_glsl.c $(LDLIBS) -lm

bin/test_chaff_multiview: tests/test_chaff_multiview.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ tests/test_chaff_multiview.c $(LDLIBS) -lm

lib/libchaff.so: src/libchaff.c include/ttypt/noise.h include/ttypt/chaff.h
	@mkdir -p lib
	$(CC) $(CFLAGS) $(SO_CFLAGS) -fPIC -shared -o $@ src/libchaff.c $(LDLIBS)
	@ln -sf libchaff.so lib/libnoise.so

wasm: npm/dist/chaff.wasm

npm/dist/chaff.wasm: npm/wasm/chaff_wasm.c include/ttypt/noise.h include/ttypt/chaff.h ../mk/wasm_alloc_shim.c
	@mkdir -p npm/dist
	$(WASM_CC) $(WASM_CFLAGS) $(WASM_LDFLAGS) npm/wasm/chaff_wasm.c ../mk/wasm_alloc_shim.c -o $@
	@ln -sf chaff.wasm npm/dist/noise.wasm

npm-build: wasm
	@mkdir -p npm/build/Release
	@gcc -O3 -fPIC -shared -I/usr/include/node -I./include -I/usr/include npm/native/chaff_napi.c -L/usr/lib -lxxhash -o npm/build/Release/chaff_node.node 2>/dev/null || true

npm-test: npm-build
	@node --test npm/tests/chaff.test.mjs

clean:
	rm -rf bin lib npm/build npm/dist/*.wasm

install: lib/libchaff.so chaff.pc
	install -d $(DESTDIR)$(PREFIX)/include/ttypt
	install -m 644 include/ttypt/noise.h $(DESTDIR)$(PREFIX)/include/ttypt/
	install -m 644 include/ttypt/chaff.h $(DESTDIR)$(PREFIX)/include/ttypt/
	install -m 644 include/ttypt/chaff_glsl.h $(DESTDIR)$(PREFIX)/include/ttypt/
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 755 lib/libchaff.so $(DESTDIR)$(PREFIX)/lib/
	ln -sf libchaff.so $(DESTDIR)$(PREFIX)/lib/libnoise.so
	install -d $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 chaff.pc $(DESTDIR)$(PREFIX)/lib/pkgconfig/

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/include/ttypt/noise.h
	rm -f $(DESTDIR)$(PREFIX)/include/ttypt/chaff.h
	rm -f $(DESTDIR)$(PREFIX)/include/ttypt/chaff_glsl.h
	rm -f $(DESTDIR)$(PREFIX)/lib/libchaff.so
	rm -f $(DESTDIR)$(PREFIX)/lib/libnoise.so
	rm -f $(DESTDIR)$(PREFIX)/lib/pkgconfig/chaff.pc

.PHONY: all test wasm npm-build npm-test install uninstall clean
