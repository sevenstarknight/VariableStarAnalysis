/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.handling.exceptions;

/**
 * <p>
 * Root finding routines may throw this exception when an error occurs in the
 * root finding routine.
 * </p>
 *
 * <p>
 * Modified by: Joseph A. Huwaldt </p>
 *
 * @author Joseph A. Huwaldt Date: October 8, 1997
 * @version July 11, 2000
 *
 */
public class RootException extends Exception {

    /**
     * Force users of this exception to supply a message by making the default
     * constructor private. A detail message is a String that describes this
     * particular exception.
     *
     */
    protected RootException() {
    }

    /**
     * Constructs a RootException with the specified detail message. A detail
     * message is a String that describes this particular exception.
     *
     * @param msg The String containing a detail message
     *
     */
    public RootException(String msg) {
        super(msg);
    }

    /**
     * Returns a short description of the RootException.
     *
     * @return Returns this exceptions message as a string.
     *
     */
    @Override
    public String toString() {
        return getMessage();
    }

}
