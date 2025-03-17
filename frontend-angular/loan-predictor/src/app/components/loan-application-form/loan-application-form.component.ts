import { Component, ElementRef, ViewChild, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-loan-application-form',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './loan-application-form.component.html',
  styleUrls: ['./loan-application-form.component.css'],
})
export class LoanApplicationFormComponent implements OnInit {
  private apiUrl = environment.apiUrl;
  dummyPayload: any = null;
  formData = {
    OCCUPATION_TYPE: '',
    AMT_INCOME_TOTAL: 0,
    AMT_CREDIT: 0,
    loanTermMonths: 0,
    EXT_SOURCE_2: 0,
  };

  dtiWarning: string = '';

  predictionResult: any = null;

  occupations: string[] = [
    'Laborers',
    'Core staff',
    'Accountants',
    'Managers',
    'Drivers',
    'Sales staff',
    'Cleaning staff',
    'Cooking staff',
    'Private service staff',
    'Medicine staff',
    'Security staff',
    'High skill tech staff',
    'Waiters/barmen staff',
    'Low-skill Laborers',
    'Realty agents',
    'Secretaries',
    'IT staff',
    'HR staff',
  ];

  @ViewChild('predictionOutput') predictionResultSection!: ElementRef;

  isLoading = false;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.http.get<any>(`${this.apiUrl}/generate_dummy`).subscribe({
      next: (response) => {
        this.dummyPayload = response;
      },
      error: (err) => {},
    });
  }

  generateDummyData() {
    this.http
      .get<any>(`${this.apiUrl}/generate_dummy`)
      .subscribe((response) => {
        this.dummyPayload = response;

        const monthlyIncome = response.application.AMT_INCOME_TOTAL ?? 0;
        let loanAmount = response.application.AMT_CREDIT ?? 0;
        const loanTermMonths =
          response.pos_cash_balance_data?.[0]?.CNT_INSTALMENT ?? 0;

        // Prevent DTI from being too high
        if (monthlyIncome > 0 && loanTermMonths > 0) {
          const maxLoanAmount = monthlyIncome * loanTermMonths * 0.3;
          if (loanAmount > maxLoanAmount) {
            loanAmount = Math.floor(maxLoanAmount);
          }
        }

        // Populate the form with the adjusted values
        this.formData = {
          OCCUPATION_TYPE: response.application.OCCUPATION_TYPE ?? '',
          AMT_INCOME_TOTAL: monthlyIncome,
          AMT_CREDIT: loanAmount,
          loanTermMonths: loanTermMonths,
          EXT_SOURCE_2: response.application.EXT_SOURCE_2 ?? 0,
        };

        this.checkDTI();
      });
  }

  predictLoan() {
    if (!this.dummyPayload) {
      return;
    }

    this.isLoading = true;

    const RUBLE_TO_USD = 90;
    const rubIncome = Math.round(this.formData.AMT_INCOME_TOTAL * RUBLE_TO_USD);
    const rubCredit = Math.round(this.formData.AMT_CREDIT * RUBLE_TO_USD);

    this.dummyPayload.application.OCCUPATION_TYPE =
      this.formData.OCCUPATION_TYPE;
    this.dummyPayload.application.AMT_INCOME_TOTAL = rubIncome;
    this.dummyPayload.application.AMT_CREDIT = rubCredit;

    if (this.dummyPayload.pos_cash_balance_data?.[0]) {
      this.dummyPayload.pos_cash_balance_data[0].CNT_INSTALMENT =
        this.formData.loanTermMonths;
    }
    this.dummyPayload.application.EXT_SOURCE_2 = this.formData.EXT_SOURCE_2;

    this.http.post<any>(`${this.apiUrl}/predict`, this.dummyPayload).subscribe(
      (response) => {
        this.predictionResult = response;
        this.isLoading = false;
        setTimeout(() => {
          this.scrollToResult();
        }, 100);
      },
      (error) => {
        this.isLoading = false;
      }
    );
  }

  scrollToResult() {
    if (this.predictionResultSection) {
      const elementY =
        this.predictionResultSection.nativeElement.getBoundingClientRect().top +
        window.scrollY;

      const offset = 600;

      window.scrollTo({
        top: elementY - offset,
        behavior: 'smooth',
      });
    }
  }

  getProbabilityNumber(): number {
    return parseFloat(this.predictionResult?.formatted_probability ?? '0');
  }

  updateNumber(event: any, field: keyof typeof this.formData) {
    let value = event.target.value.replace(/,/g, '');
    if (!isNaN(value) && field in this.formData) {
      (this.formData as any)[field] = Number(value);
    }

    // Check if we need to show a DTI warning
    this.checkDTI();
  }

  checkDTI() {
    const monthlyIncome = this.formData.AMT_INCOME_TOTAL;
    const loanAmount = this.formData.AMT_CREDIT;
    const loanTermMonths = this.formData.loanTermMonths;

    if (monthlyIncome > 0 && loanAmount > 0 && loanTermMonths > 0) {
      const estimatedAnnuity = loanAmount / loanTermMonths;
      const dtiRatio = estimatedAnnuity / monthlyIncome;

      if (dtiRatio > 0.3) {
        this.dtiWarning = `DTI ratio too high (${(dtiRatio * 100).toFixed(
          1
        )}%). Please choose a lower loan amount.`;
      } else {
        this.dtiWarning = '';
      }
    }
  }

  isDTITooHigh(): boolean {
    const monthlyIncome = this.formData.AMT_INCOME_TOTAL;
    const loanAmount = this.formData.AMT_CREDIT;
    const loanTermMonths = this.formData.loanTermMonths;

    if (monthlyIncome > 0 && loanAmount > 0 && loanTermMonths > 0) {
      const estimatedAnnuity = loanAmount / loanTermMonths;
      const dtiRatio = estimatedAnnuity / monthlyIncome;

      return dtiRatio > 0.3;
    }
    return false;
  }
}
