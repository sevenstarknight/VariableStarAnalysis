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
package fit.astro.vsa.common.utilities.math.handling.sigfig;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SignificantDigits {

    // Empty Constructor
    private SignificantDigits() {

    }

    /**
     *
     * @param d
     * <p>
     * @param numberOfDigits
     * @return
     */
    public static double roundSignificantDigitsError(double d, int numberOfDigits) {

        BigDecimal bd = BigDecimal.valueOf(d);
        bd = bd.round(new MathContext(numberOfDigits));
        return bd.doubleValue();
    }

    /**
     *
     * @param value
     * @param error
     * <p>
     * @return
     */
    public static double roundToPrecision(double value, double error) {

        BigDecimal errorBig = new BigDecimal(String.valueOf(error));
        int scale = errorBig.scale();

        BigDecimal valueBig = BigDecimal.valueOf(value);
        return valueBig.setScale(scale,
                RoundingMode.HALF_DOWN).doubleValue();

    }

}
