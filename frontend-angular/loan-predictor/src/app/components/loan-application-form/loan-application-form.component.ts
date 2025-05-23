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
    MONTHLY_INCOME: 0,
    LOAN_AMOUNT: 0,
    loanTermMonths: 0,
    EXTERNAL_CREDIT_SCORE: 0,
  };

  dtiWarning: string = '';

  predictionResult: any = null;

  occupations: string[] = [];

  categoryMappings: any = {};

  @ViewChild('predictionOutput') predictionResultSection!: ElementRef;

  isLoading = false;

  private predictTimer!: any;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.http.get<any>(`${this.apiUrl}/generate_dummy`).subscribe({
      next: (response) => {
        this.dummyPayload = response;
      },
      error: (err) => {},
    });

    this.http.get<any>(`${this.apiUrl}/feature_metadata`).subscribe({
      next: (meta) => {
        this.categoryMappings = meta.category_mappings || {};
        this.occupations = this.categoryMappings['OCCUPATION_TYPE'] || [];
      },
      error: (err) => {
        console.error('Could not load metadata', err);
      },
    });
  }

  generateDummyData() {
    this.http
      .get<any>(`${this.apiUrl}/generate_dummy`)
      .subscribe((response) => {
        this.dummyPayload = response;

        const monthlyIncome = response.customer_profile.MONTHLY_INCOME ?? 0;
        let loanAmount = response.customer_profile.LOAN_AMOUNT ?? 0;

        const loanTermMonths = response.customer_profile.LOAN_TERM_MONTHS ?? 0;

        // Cap DTI
        if (monthlyIncome > 0 && loanTermMonths > 0) {
          const maxLoan = monthlyIncome * loanTermMonths * 0.3;
          if (loanAmount > maxLoan) {
            loanAmount = Math.floor(maxLoan);
          }
        }

        this.formData = {
          OCCUPATION_TYPE: response.customer_profile.OCCUPATION_TYPE,
          MONTHLY_INCOME: monthlyIncome,
          LOAN_AMOUNT: loanAmount,
          loanTermMonths: loanTermMonths,
          EXTERNAL_CREDIT_SCORE:
            response.customer_profile.EXTERNAL_CREDIT_SCORE ?? 0,
        };

        this.checkDTI();
      });
  }

  triggerPredict() {
    if (this.isDTITooHigh()) {
      return;
    }
    clearTimeout(this.predictTimer);
    this.predictTimer = setTimeout(() => this.predictLoan(), 300);
  }

  predictLoan() {
    if (!this.dummyPayload) {
      return;
    }

    this.isLoading = true;

    const usdIncome = this.formData.MONTHLY_INCOME;
    const usdCredit = this.formData.LOAN_AMOUNT;

    this.dummyPayload.customer_profile.OCCUPATION_TYPE =
      this.formData.OCCUPATION_TYPE;
    this.dummyPayload.customer_profile.MONTHLY_INCOME = usdIncome;
    this.dummyPayload.customer_profile.LOAN_AMOUNT = usdCredit;

    if (this.dummyPayload.cash_pos_records?.[0]) {
      this.dummyPayload.cash_pos_records[0].LOAN_TERM_MONTHS =
        this.formData.loanTermMonths;
    }
    this.dummyPayload.customer_profile.EXTERNAL_CREDIT_SCORE =
      this.formData.EXTERNAL_CREDIT_SCORE;

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

    this.checkDTI();
    this.triggerPredict();
  }

  checkDTI() {
    const monthlyIncome = this.formData.MONTHLY_INCOME;
    const loanAmount = this.formData.LOAN_AMOUNT;
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
    const monthlyIncome = this.formData.MONTHLY_INCOME;
    const loanAmount = this.formData.LOAN_AMOUNT;
    const loanTermMonths = this.formData.loanTermMonths;

    if (monthlyIncome > 0 && loanAmount > 0 && loanTermMonths > 0) {
      const estimatedAnnuity = loanAmount / loanTermMonths;
      const dtiRatio = estimatedAnnuity / monthlyIncome;

      return dtiRatio > 0.3;
    }
    return false;
  }
}
