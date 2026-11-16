all := libchaff
SONAME-libchaff := libnoise
LDLIBS-libchaff :=

WASM_SRCS := npm/wasm/chaff_wasm.c

include ../mk/include.mk
LDLIBS-alpine :=

CFLAGS := -I./include -O2 -Wno-overlength-strings ${CFLAGS} $(EXTRA_CFLAGS)

TEST_BINS := bin/test_noise${EXE} bin/test_chaff${EXE} bin/test_symbols${EXE} bin/test_glsl${EXE} bin/test_chaff_multiview${EXE}
BENCH_BINS := bin/bench_noise${EXE}

sync-npm-headers:
	@install -d npm/include/ttypt
	@install -m 644 include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h npm/include/ttypt/

all: sync-npm-headers

wasm: sync-npm-headers ${wasmtarget}
	@install -m 644 npm/dist/chaff.wasm npm/dist/noise.wasm

test: all $(TEST_BINS)
	./bin/test_noise${EXE}
	./bin/test_chaff${EXE}
	./bin/test_symbols${EXE}
	./bin/test_glsl${EXE}
	./bin/test_chaff_multiview${EXE}

bench: $(BENCH_BINS)
	./bin/bench_noise${EXE}

bin/bench_noise${EXE}: tests/bench_noise.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	${cc} ${CFLAGS} -o $@ tests/bench_noise.c -lm

bin/test_noise${EXE}: tests/test_noise.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	${cc} ${CFLAGS} -o $@ tests/test_noise.c -lm

bin/test_chaff${EXE}: tests/test_chaff.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	${cc} ${CFLAGS} -o $@ tests/test_chaff.c -lm

bin/test_symbols${EXE}: tests/test_symbols.c src/libchaff.c include/ttypt/noise.h include/ttypt/chaff.h
	@mkdir -p bin
	${cc} ${CFLAGS} -o $@ tests/test_symbols.c src/libchaff.c -lm

bin/test_glsl${EXE}: tests/test_glsl.c include/ttypt/chaff_glsl.h
	@mkdir -p bin
	${cc} ${CFLAGS} -o $@ tests/test_glsl.c -lm

bin/test_chaff_multiview${EXE}: tests/test_chaff_multiview.c include/ttypt/noise.h include/ttypt/chaff.h include/ttypt/chaff_glsl.h
	@mkdir -p bin
	${cc} ${CFLAGS} -o $@ tests/test_chaff_multiview.c -lm

${DESTDIR}${PREFIX}/lib/pkgconfig/chaff.pc: chaff.pc
	install -d ${DESTDIR}${PREFIX}/lib/pkgconfig
	install -m 644 chaff.pc $@

install: ${DESTDIR}${PREFIX}/lib/pkgconfig/chaff.pc

NPM ?= npm

npm-build: wasm
	@if command -v $(NPM) >/dev/null 2>&1; then \
		(cd npm && $(NPM) run build); \
	fi

npm-test: npm-build
	@if command -v $(NPM) >/dev/null 2>&1; then \
		(cd npm && $(NPM) test); \
	fi

clean-extra::
	rm -rf bin npm/build npm/dist/*.wasm

objects-set.mk: Makefile

.PHONY: all sync-npm-headers test bench wasm npm-build npm-test install
