#!/bin/bash
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## This file is adapted by Sambit Das for use with dft-efe code
## ---------------------------------------------------------------------

#
# This script downloads and installs the clang-format binary. The
# destination directory is
#   [contrib/utilities]/programs/clang-<VERSION>/bin.
#
# This script only works on Linux (amd64) and macOS. For other
# architectures it is necessary to compile the clang-format binary by hand.
# This can be done with the compile_clang_format script.
#

VERSION=11
PRG="$(cd "$(dirname "$0")" && pwd)/programs"
CLANG_PATH="${PRG}/clang-${VERSION}"

# Find out which kind of OS we are running and set the appropriate settings
case "${OSTYPE}" in
  linux*)
    FILENAME="clang-format-${VERSION}-linux.tar.gz"
    CHECKSUM_CMD="sha256sum"
    CHECKSUM="9420c4ed80268500ef357eb42b2e77dc55e807e559d6d9851f4808a9939fa47b  $FILENAME"
    ;;
  darwin*)
    FILENAME="clang-format-${VERSION}-darwin-intel.tar.gz"
    CHECKSUM_CMD="shasum"
    CHECKSUM="5b8c310a660102a1aa46cc0242294fb14271797a4027883a698651411f8e51bf  $FILENAME"
    ;;
  *)
    echo "unknown: ${OSTYPE}"
    exit 1
    ;;
esac

if [ ! -d "${PRG}" ]
then
    echo "create folder ${PRG}"
    mkdir "${PRG}"
fi

if [ -d "${CLANG_PATH}" ]
then
    echo "${CLANG_PATH}  exists. Exiting."
    exit 1
fi

echo "Installing clang-format-${VERSION} from ${FILENAME}"
mkdir "${CLANG_PATH}"

tmpdir="${TMPDIR:-/tmp}/dftefeclang${RANDOM}${RANDOM}"
mkdir -p "${tmpdir}"
cd "${tmpdir}"
cp "${PRG}/../$FILENAME" .

if echo "${CHECKSUM}" | "${CHECKSUM_CMD}" -c; then
  tar xfz "${FILENAME}" -C "${PRG}" > /dev/null
else
  echo "*** The downloaded file has the wrong SHA256 checksum!"
  exit 1
fi
rm -r "${tmpdir}"

echo "All done. clang-format successfully installed into"
echo "    ${CLANG_PATH}/bin"
