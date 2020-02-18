/*!************************************************************************
 * Provides a message stack to print more useful error
 * messages.
 *
 **************************************************************************
 * Copyright 2010 B.D.Dudson, S.Farley, M.V.Umansky, X.Q.Xu
 *
 * Contact: Ben Dudson, bd512@york.ac.uk
 *
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#include "bout/openmpwrap.hxx"
#include <msg_stack.hxx>
#include <output.hxx>
#include <cstdarg>
#include <string>

#if CHECK > 1
int MsgStack::push(std::string message) {
  BOUT_OMP(critical(MsgStack)) {
    if (position >= stack.size()) {
      stack.push_back(std::move(message));
    } else {
      stack[position] = message;
    }

    position++;
  };
  int result;
  BOUT_OMP(critical(MsgStack))
  result = position - 1;
  return result;
}

int MsgStack::setPoint() {
  // Create an empty message
  return push();
}

void MsgStack::pop() {
  if (position <= 0)
    return;
  BOUT_OMP(atomic)
  --position;
}

void MsgStack::pop(int id) {
  if (id < 0)
    id = 0;

  BOUT_OMP(critical(MsgStack)) {
    if (id <= static_cast<int>(position))
      position = id;
  };
}

void MsgStack::clear() {
  BOUT_OMP(single) {
    stack.clear();
    position = 0;
  }
}

void MsgStack::dump() {
  BOUT_OMP(single) { output << this->getDump(); }
}

std::string MsgStack::getDump() {
  std::string res = "====== Back trace ======\n";
  for (int i = position - 1; i >= 0; i--) {
    if (stack[i] != "") {
      res += " -> ";
      res += stack[i];
      res += "\n";
    }
  }
  return res;
}

#endif
